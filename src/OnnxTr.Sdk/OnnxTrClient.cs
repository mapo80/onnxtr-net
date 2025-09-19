using System.Buffers;
using System.Diagnostics;
using System.Linq;
using System.Net.Http;
using System.Numerics.Tensors;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxTr.Sdk.Results;
using SkiaSharp;

namespace OnnxTr.Sdk;

/// <summary>
/// Provides a native .NET façade over the OnnxTR OCR models using Microsoft Onnx Runtime.
/// </summary>
public sealed class OnnxTrClient : IDisposable
{
    private const float LineGroupingTolerance = 0.04f;

    private static readonly HttpClient HttpClient = new();

    private static readonly IReadOnlyDictionary<OnnxTrDetectionModel, DetectionModelConfig> DetectionModels =
        new Dictionary<OnnxTrDetectionModel, DetectionModelConfig>
        {
            [OnnxTrDetectionModel.FastTiny] = new(
                "fast_tiny",
                new Uri("https://github.com/felixdittrich92/OnnxTR/releases/download/v0.0.1/rep_fast_tiny-28867779.onnx"),
                1024,
                1024,
                new[] { 0.798f, 0.785f, 0.772f },
                new[] { 0.264f, 0.2749f, 0.287f },
                0.1f,
                0.1f),
            [OnnxTrDetectionModel.FastSmall] = new(
                "fast_small",
                new Uri("https://github.com/felixdittrich92/OnnxTR/releases/download/v0.0.1/rep_fast_small-10428b70.onnx"),
                1024,
                1024,
                new[] { 0.798f, 0.785f, 0.772f },
                new[] { 0.264f, 0.2749f, 0.287f },
                0.1f,
                0.1f),
            [OnnxTrDetectionModel.FastBase] = new(
                "fast_base",
                new Uri("https://github.com/felixdittrich92/OnnxTR/releases/download/v0.0.1/rep_fast_base-1b89ebf9.onnx"),
                1024,
                1024,
                new[] { 0.798f, 0.785f, 0.772f },
                new[] { 0.264f, 0.2749f, 0.287f },
                0.1f,
                0.1f),
        };

    private static readonly string FrenchVocabulary =
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~°£€¥¢฿"
        + "àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ";

    private static readonly IReadOnlyDictionary<OnnxTrRecognitionModel, RecognitionModelConfig> RecognitionModels =
        new Dictionary<OnnxTrRecognitionModel, RecognitionModelConfig>
        {
            [OnnxTrRecognitionModel.CrnnVgg16Bn] = new(
                "crnn_vgg16_bn",
                new Uri("https://github.com/felixdittrich92/OnnxTR/releases/download/v0.7.1/crnn_vgg16_bn-743599aa.onnx"),
                32,
                128,
                new[] { 0.694f, 0.695f, 0.693f },
                new[] { 0.299f, 0.296f, 0.301f },
                FrenchVocabulary),
            [OnnxTrRecognitionModel.CrnnMobilenetV3Small] = new(
                "crnn_mobilenet_v3_small",
                new Uri("https://github.com/felixdittrich92/OnnxTR/releases/download/v0.0.1/crnn_mobilenet_v3_small-bded4d49.onnx"),
                32,
                128,
                new[] { 0.694f, 0.695f, 0.693f },
                new[] { 0.299f, 0.296f, 0.301f },
                FrenchVocabulary),
            [OnnxTrRecognitionModel.CrnnMobilenetV3Large] = new(
                "crnn_mobilenet_v3_large",
                new Uri("https://github.com/felixdittrich92/OnnxTR/releases/download/v0.0.1/crnn_mobilenet_v3_large-d42e8185.onnx"),
                32,
                128,
                new[] { 0.694f, 0.695f, 0.693f },
                new[] { 0.299f, 0.296f, 0.301f },
                FrenchVocabulary),
        };

    private readonly OnnxTrBackend _backend;
    private readonly OnnxTrClientOptions _options;
    private readonly OnnxTrRunOptions _runOptions;
    private readonly DetectionModelConfig _detectionConfig;
    private readonly RecognitionModelConfig _recognitionConfig;
    private readonly SemaphoreSlim _modelGate = new(1, 1);

    private InferenceSession? _detectionSession;
    private InferenceSession? _recognitionSession;
    private string? _detectionInputName;
    private string? _detectionOutputName;
    private string? _recognitionInputName;
    private string? _recognitionOutputName;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the <see cref="OnnxTrClient"/> class.
    /// </summary>
    public OnnxTrClient(OnnxTrBackend backend, OnnxTrClientOptions? options = null, OnnxTrRunOptions? runOptions = null)
    {
        _backend = backend;
        _options = options ?? new OnnxTrClientOptions();
        _runOptions = runOptions ?? new OnnxTrRunOptions();

        if (_runOptions.DetectOrientation)
        {
            throw new NotSupportedException("Orientation detection is not supported by the native .NET OnnxTR client.");
        }

        if (_runOptions.DetectLanguage)
        {
            throw new NotSupportedException("Language detection is not supported by the native .NET OnnxTR client.");
        }

        if (_runOptions.StraightenPages)
        {
            throw new NotSupportedException("Page straightening is not supported by the native .NET OnnxTR client.");
        }

        _detectionConfig = ResolveDetectionConfig(_runOptions.DetectionModel);
        _recognitionConfig = ResolveRecognitionConfig(_runOptions.RecognitionModel);
    }

    /// <summary>
    /// Runs OCR on a collection of images.
    /// </summary>
    public async Task<OnnxTrResult> PredictAsync(IEnumerable<OnnxTrImageInput> images, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(images);

        var materialised = images.ToList();
        if (materialised.Count == 0)
        {
            throw new ArgumentException("At least one image must be provided.", nameof(images));
        }

        await EnsureSessionsAsync(cancellationToken).ConfigureAwait(false);

        var pages = new List<OnnxTrPage>(materialised.Count);
        var aggregatedLines = new List<string>();

        for (var index = 0; index < materialised.Count; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var page = await ProcessPageAsync(materialised[index], index, cancellationToken).ConfigureAwait(false);
            pages.Add(page);
            aggregatedLines.AddRange(CollectTextLines(page));
        }

        var document = new OnnxTrDocument { Pages = pages };
        var text = string.Join(Environment.NewLine, aggregatedLines.Where(line => !string.IsNullOrWhiteSpace(line)));

        return new OnnxTrResult
        {
            Backend = _backend.ToString(),
            Document = document,
            Text = text,
        };
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _detectionSession?.Dispose();
        _recognitionSession?.Dispose();
        _modelGate.Dispose();
    }

    private async Task EnsureSessionsAsync(CancellationToken cancellationToken)
    {
        if (_detectionSession is not null && _recognitionSession is not null)
        {
            return;
        }

        await _modelGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (_detectionSession is null)
            {
                var detectionModelPath = await ResolveModelPathAsync(
                    _options.DetectionModelPath,
                    _detectionConfig.SourceUri,
                    cancellationToken).ConfigureAwait(false);

                _detectionSession = CreateSession(detectionModelPath);
                var detectionInput = _detectionSession.InputMetadata.Single();
                _detectionInputName = detectionInput.Key;
                var detectionOutput = _detectionSession.OutputMetadata.Single();
                _detectionOutputName = detectionOutput.Key;
            }

            if (_recognitionSession is null)
            {
                var recognitionModelPath = await ResolveModelPathAsync(
                    _options.RecognitionModelPath,
                    _recognitionConfig.SourceUri,
                    cancellationToken).ConfigureAwait(false);

                _recognitionSession = CreateSession(recognitionModelPath);
                var recognitionInput = _recognitionSession.InputMetadata.Single();
                _recognitionInputName = recognitionInput.Key;
                var recognitionOutput = _recognitionSession.OutputMetadata.Single();
                _recognitionOutputName = recognitionOutput.Key;
            }
        }
        finally
        {
            _modelGate.Release();
        }
    }

    private async Task<OnnxTrPage> ProcessPageAsync(OnnxTrImageInput input, int pageIndex, CancellationToken cancellationToken)
    {
        using var image = await LoadImageAsync(input, cancellationToken).ConfigureAwait(false);
        var originalWidth = image.Width;
        var originalHeight = image.Height;

        var detectionInput = PrepareDetectionInput(image);
        var probabilityMap = RunDetection(detectionInput.Tensor);
        var boxes = ExtractBoxes(probabilityMap, detectionInput.Metadata);

        var wordPredictions = new List<WordPrediction>();

        foreach (var box in boxes)
        {
            cancellationToken.ThrowIfCancellationRequested();

            using var crop = ExtractCrop(image, box, originalWidth, originalHeight);
            if (crop is null)
            {
                continue;
            }

            var recognitionInput = PrepareRecognitionInput(crop);
            var (text, confidence) = RunRecognition(recognitionInput);

            if (string.IsNullOrWhiteSpace(text))
            {
                continue;
            }

            wordPredictions.Add(new WordPrediction(text, confidence, box, box.Score));
        }

        var page = BuildPage(pageIndex, originalWidth, originalHeight, wordPredictions);
        return page;
    }

    private DenseTensor<float> PrepareRecognitionInput(SKBitmap crop)
    {
        var config = _recognitionConfig;
        var (tensor, _) = ResizeAndNormalize(crop, config.InputWidth, config.InputHeight, config.Mean, config.Std, includeMetadata: false);
        return tensor;
    }

    private DetectionInput PrepareDetectionInput(SKBitmap image)
    {
        var config = _detectionConfig;
        var (tensor, metadata) = ResizeAndNormalize(image, config.InputWidth, config.InputHeight, config.Mean, config.Std, includeMetadata: true);
        return new DetectionInput(tensor, metadata);
    }

    private float[,] RunDetection(DenseTensor<float> inputTensor)
    {
        Debug.Assert(_detectionSession is not null);
        Debug.Assert(_detectionInputName is not null);
        Debug.Assert(_detectionOutputName is not null);

        var input = NamedOnnxValue.CreateFromTensor(_detectionInputName!, inputTensor);
        try
        {
            using var results = _detectionSession!.Run(new[] { input });
            using var output = results.First();
            var tensor = output.AsTensor<float>();
            if (tensor.Dimensions.Length != 4)
            {
                throw new InvalidOperationException("Unexpected detection output shape.");
            }

            var height = tensor.Dimensions[2];
            var width = tensor.Dimensions[3];
            var map = new float[height, width];
            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    map[y, x] = Sigmoid(tensor[0, 0, y, x]);
                }
            }

            return map;
        }
        finally
        {
            if (input is IDisposable disposable)
            {
                disposable.Dispose();
            }
        }
    }

    private (string Text, float Confidence) RunRecognition(DenseTensor<float> inputTensor)
    {
        Debug.Assert(_recognitionSession is not null);
        Debug.Assert(_recognitionInputName is not null);
        Debug.Assert(_recognitionOutputName is not null);

        var input = NamedOnnxValue.CreateFromTensor(_recognitionInputName!, inputTensor);
        try
        {
            using var results = _recognitionSession!.Run(new[] { input });
            using var output = results.First();
            var tensor = output.AsTensor<float>();
            if (tensor.Dimensions.Length != 3)
            {
                throw new InvalidOperationException("Unexpected recognition output shape.");
            }

            var batch = tensor.Dimensions[0];
            if (batch != 1)
            {
                throw new InvalidOperationException("Recognition output batch size must be 1.");
            }

            var vocabLength = _recognitionConfig.Vocabulary.Length;
            var classCount = vocabLength + 1;

            bool classesLast;
            int timeSteps;
            if (tensor.Dimensions[2] == classCount)
            {
                classesLast = true;
                timeSteps = tensor.Dimensions[1];
            }
            else if (tensor.Dimensions[1] == classCount)
            {
                classesLast = false;
                timeSteps = tensor.Dimensions[2];
            }
            else
            {
                throw new InvalidOperationException("Unable to resolve recognition output layout.");
            }

            using var owner = MemoryPool<float>.Shared.Rent(classCount);
            var buffer = owner.Memory.Span.Slice(0, classCount);

            var builder = new System.Text.StringBuilder();
            var minProbability = 1f;
            var blankIndex = vocabLength;
            int? previousIndex = null;

            for (var t = 0; t < timeSteps; t++)
            {
                for (var c = 0; c < classCount; c++)
                {
                    buffer[c] = classesLast ? tensor[0, t, c] : tensor[0, c, t];
                }

                var (index, probability) = ArgMaxWithProbability(buffer);
                if (probability < minProbability)
                {
                    minProbability = probability;
                }

                if (index == blankIndex)
                {
                    previousIndex = null;
                    continue;
                }

                if (previousIndex.HasValue && previousIndex.Value == index)
                {
                    continue;
                }

                if (index >= 0 && index < vocabLength)
                {
                    builder.Append(_recognitionConfig.Vocabulary[index]);
                }

                previousIndex = index;
            }

            if (builder.Length == 0)
            {
                minProbability = 0f;
            }

            return (builder.ToString(), minProbability);
        }
        finally
        {
            if (input is IDisposable disposable)
            {
                disposable.Dispose();
            }
        }
    }

    private OnnxTrPage BuildPage(int pageIndex, int width, int height, List<WordPrediction> words)
    {
        var page = new OnnxTrPage
        {
            PageIndex = pageIndex,
            Dimensions = new[] { (float)height, (float)width },
        };

        if (words.Count == 0)
        {
            return page;
        }

        var lines = GroupIntoLines(words);
        var blockLines = new List<OnnxTrLine>();
        foreach (var lineWords in lines)
        {
            if (lineWords.Count == 0)
            {
                continue;
            }

            var line = new OnnxTrLine
            {
                Geometry = BuildGeometry(lineWords),
                ObjectnessScore = lineWords.Average(w => w.ObjectnessScore),
            };

            foreach (var word in lineWords)
            {
                line.Words.Add(new OnnxTrWord
                {
                    Value = word.Text,
                    Confidence = word.Confidence,
                    ObjectnessScore = word.ObjectnessScore,
                    Geometry = BuildGeometry(word.Box),
                });
            }

            blockLines.Add(line);
        }

        if (blockLines.Count > 0)
        {
            var block = new OnnxTrBlock
            {
                Lines = blockLines,
                ObjectnessScore = blockLines
                    .SelectMany(line => line.Words)
                    .Select(word => word.ObjectnessScore ?? 0f)
                    .DefaultIfEmpty(0f)
                    .Average(),
                Geometry = BuildGeometry(words),
            };

            page.Blocks.Add(block);
        }

        return page;
    }

    private static List<List<WordPrediction>> GroupIntoLines(List<WordPrediction> words)
    {
        var ordered = words
            .OrderBy(w => w.Box.YMin)
            .ThenBy(w => w.Box.XMin)
            .ToList();

        var lines = new List<List<WordPrediction>>();

        foreach (var word in ordered)
        {
            var centerY = (word.Box.YMin + word.Box.YMax) / 2f;
            if (lines.Count == 0)
            {
                lines.Add(new List<WordPrediction> { word });
                continue;
            }

            var currentLine = lines[^1];
            var currentCenter = currentLine.Average(w => (w.Box.YMin + w.Box.YMax) / 2f);

            if (MathF.Abs(centerY - currentCenter) <= LineGroupingTolerance)
            {
                currentLine.Add(word);
            }
            else
            {
                lines.Add(new List<WordPrediction> { word });
            }
        }

        foreach (var line in lines)
        {
            line.Sort((left, right) => left.Box.XMin.CompareTo(right.Box.XMin));
        }

        return lines;
    }

    private List<NormalizedBox> ExtractBoxes(float[,] map, DetectionMetadata metadata)
    {
        var height = map.GetLength(0);
        var width = map.GetLength(1);
        var visited = new bool[height, width];
        var boxes = new List<NormalizedBox>();

        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                if (visited[y, x] || map[y, x] < _detectionConfig.BinThreshold)
                {
                    continue;
                }

                var score = FloodFill(map, visited, x, y, metadata);
                if (score is not null)
                {
                    boxes.Add(score.Value);
                }
            }
        }

        return boxes;
    }

    private NormalizedBox? FloodFill(float[,] map, bool[,] visited, int startX, int startY, DetectionMetadata metadata)
    {
        var height = map.GetLength(0);
        var width = map.GetLength(1);
        var queue = new Queue<(int X, int Y)>();
        queue.Enqueue((startX, startY));
        visited[startY, startX] = true;

        var minX = startX;
        var maxX = startX;
        var minY = startY;
        var maxY = startY;
        var sum = 0f;
        var count = 0;

        while (queue.Count > 0)
        {
            var (x, y) = queue.Dequeue();
            var value = map[y, x];
            sum += value;
            count++;

            if (x < minX)
            {
                minX = x;
            }
            if (x > maxX)
            {
                maxX = x;
            }
            if (y < minY)
            {
                minY = y;
            }
            if (y > maxY)
            {
                maxY = y;
            }

            EnqueueIfValid(x - 1, y);
            EnqueueIfValid(x + 1, y);
            EnqueueIfValid(x, y - 1);
            EnqueueIfValid(x, y + 1);
        }

        if (count == 0)
        {
            return null;
        }

        if (maxX - minX < 2 || maxY - minY < 2)
        {
            return null;
        }

        var score = sum / count;
        if (score < _detectionConfig.BoxThreshold)
        {
            return null;
        }

        var mapped = MapToOriginal(minX, minY, maxX, maxY, width, height, metadata, score);
        return mapped;

        void EnqueueIfValid(int x, int y)
        {
            if (x < 0 || y < 0 || x >= width || y >= height)
            {
                return;
            }

            if (visited[y, x] || map[y, x] < _detectionConfig.BinThreshold)
            {
                return;
            }

            visited[y, x] = true;
            queue.Enqueue((x, y));
        }
    }

    private static NormalizedBox MapToOriginal(
        int minX,
        int minY,
        int maxX,
        int maxY,
        int outputWidth,
        int outputHeight,
        DetectionMetadata metadata,
        float score)
    {
        var paddedWidth = metadata.TargetWidth;
        var paddedHeight = metadata.TargetHeight;

        var xMinPad = (minX / (float)outputWidth) * paddedWidth;
        var xMaxPad = ((maxX + 1) / (float)outputWidth) * paddedWidth;
        var yMinPad = (minY / (float)outputHeight) * paddedHeight;
        var yMaxPad = ((maxY + 1) / (float)outputHeight) * paddedHeight;

        var xMin = (xMinPad - metadata.OffsetX) / metadata.ScaleX;
        var xMax = (xMaxPad - metadata.OffsetX) / metadata.ScaleX;
        var yMin = (yMinPad - metadata.OffsetY) / metadata.ScaleY;
        var yMax = (yMaxPad - metadata.OffsetY) / metadata.ScaleY;

        var xMinNorm = Clamp01(xMin / metadata.OriginalWidth);
        var xMaxNorm = Clamp01(xMax / metadata.OriginalWidth);
        if (xMinNorm > xMaxNorm)
        {
            (xMinNorm, xMaxNorm) = (xMaxNorm, xMinNorm);
        }

        var yMinNorm = Clamp01(yMin / metadata.OriginalHeight);
        var yMaxNorm = Clamp01(yMax / metadata.OriginalHeight);
        if (yMinNorm > yMaxNorm)
        {
            (yMinNorm, yMaxNorm) = (yMaxNorm, yMinNorm);
        }

        return new NormalizedBox(xMinNorm, yMinNorm, xMaxNorm, yMaxNorm, score);
    }

    private static SKBitmap? ExtractCrop(SKBitmap image, NormalizedBox box, int width, int height)
    {
        var left = Math.Clamp((int)MathF.Floor(box.XMin * width), 0, width - 1);
        var top = Math.Clamp((int)MathF.Floor(box.YMin * height), 0, height - 1);
        var right = Math.Clamp((int)MathF.Ceiling(box.XMax * width), left + 1, width);
        var bottom = Math.Clamp((int)MathF.Ceiling(box.YMax * height), top + 1, height);

        var cropWidth = right - left;
        var cropHeight = bottom - top;

        if (cropWidth <= 0 || cropHeight <= 0)
        {
            return null;
        }

        var crop = new SKBitmap(cropWidth, cropHeight);
        using var canvas = new SKCanvas(crop);
        var source = new SKRectI(left, top, right, bottom);
        var destination = new SKRect(0, 0, cropWidth, cropHeight);
        canvas.DrawBitmap(image, source, destination);
        canvas.Flush();
        return crop;
    }

    private (DenseTensor<float> Tensor, DetectionMetadata Metadata) ResizeAndNormalize(
        SKBitmap image,
        int targetWidth,
        int targetHeight,
        IReadOnlyList<float> mean,
        IReadOnlyList<float> std,
        bool includeMetadata)
    {
        var scale = Math.Min((float)targetWidth / image.Width, (float)targetHeight / image.Height);
        var scaledWidth = Math.Max(1, (int)MathF.Round(image.Width * scale));
        var scaledHeight = Math.Max(1, (int)MathF.Round(image.Height * scale));
        var offsetX = Math.Max(0, (targetWidth - scaledWidth) / 2);
        var offsetY = Math.Max(0, (targetHeight - scaledHeight) / 2);

        using var resized = image.Resize(new SKImageInfo(scaledWidth, scaledHeight, SKColorType.Rgba8888, SKAlphaType.Premul), SKFilterQuality.High)
            ?? throw new InvalidOperationException("Failed to resize bitmap for preprocessing.");

        using var canvasBitmap = new SKBitmap(new SKImageInfo(targetWidth, targetHeight, SKColorType.Rgba8888, SKAlphaType.Premul));
        using (var canvas = new SKCanvas(canvasBitmap))
        {
            canvas.Clear(SKColors.Black);
            var destination = new SKRect(offsetX, offsetY, offsetX + scaledWidth, offsetY + scaledHeight);
            canvas.DrawBitmap(resized, destination);
            canvas.Flush();
        }

        var tensor = new DenseTensor<float>(new[] { 1, 3, targetHeight, targetWidth });

        for (var y = 0; y < targetHeight; y++)
        {
            for (var x = 0; x < targetWidth; x++)
            {
                var pixel = canvasBitmap.GetPixel(x, y);
                tensor[0, 0, y, x] = (pixel.Red / 255f - mean[0]) / std[0];
                tensor[0, 1, y, x] = (pixel.Green / 255f - mean[1]) / std[1];
                tensor[0, 2, y, x] = (pixel.Blue / 255f - mean[2]) / std[2];
            }
        }

        var metadata = includeMetadata
            ? new DetectionMetadata(
                image.Width,
                image.Height,
                targetWidth,
                targetHeight,
                Math.Max(scale, 1e-6f),
                Math.Max(scale, 1e-6f),
                offsetX,
                offsetY)
            : default;

        return (tensor, metadata);
    }

    private async Task<SKBitmap> LoadImageAsync(OnnxTrImageInput input, CancellationToken cancellationToken)
    {
        if (input.FilePath is not null)
        {
            var fromFile = SKBitmap.Decode(input.FilePath)
                ?? throw new InvalidOperationException($"Unable to decode image at path '{input.FilePath}'.");
            return fromFile;
        }

        ArgumentNullException.ThrowIfNull(input.Content);
        if (input.Content.CanSeek)
        {
            input.Content.Seek(0, SeekOrigin.Begin);
        }

        using var memory = new MemoryStream();
        await input.Content.CopyToAsync(memory, cancellationToken).ConfigureAwait(false);
        memory.Position = 0;
        var bitmap = SKBitmap.Decode(memory.ToArray())
            ?? throw new InvalidOperationException("Unable to decode image from provided stream.");
        return bitmap;
    }

    private static List<IReadOnlyList<float>>? BuildGeometry(List<WordPrediction> words)
    {
        if (words.Count == 0)
        {
            return null;
        }

        var minX = words.Min(w => w.Box.XMin);
        var minY = words.Min(w => w.Box.YMin);
        var maxX = words.Max(w => w.Box.XMax);
        var maxY = words.Max(w => w.Box.YMax);

        return new List<IReadOnlyList<float>>
        {
            new[] { minX, minY },
            new[] { maxX, maxY },
        };
    }

    private static List<IReadOnlyList<float>> BuildGeometry(NormalizedBox box)
    {
        return new List<IReadOnlyList<float>>
        {
            new[] { box.XMin, box.YMin },
            new[] { box.XMax, box.YMax },
        };
    }

    private static IEnumerable<string> CollectTextLines(OnnxTrPage page)
    {
        foreach (var block in page.Blocks)
        {
            foreach (var line in block.Lines)
            {
                var text = string.Join(
                    " ",
                    line.Words
                        .Select(word => word.Value)
                        .Where(value => !string.IsNullOrWhiteSpace(value)));

                if (!string.IsNullOrWhiteSpace(text))
                {
                    yield return text;
                }
            }
        }
    }

    private DetectionModelConfig ResolveDetectionConfig(OnnxTrDetectionModel model)
    {
        if (!DetectionModels.TryGetValue(model, out var config))
        {
            throw new ArgumentOutOfRangeException(nameof(model), model, "Unsupported detection model.");
        }

        return config;
    }

    private RecognitionModelConfig ResolveRecognitionConfig(OnnxTrRecognitionModel model)
    {
        if (!RecognitionModels.TryGetValue(model, out var config))
        {
            throw new ArgumentOutOfRangeException(nameof(model), model, "Unsupported recognition model.");
        }

        return config;
    }

    private InferenceSession CreateSession(string modelPath)
    {
        var options = new SessionOptions();

        options.AppendExecutionProvider_CPU();
        if (_backend == OnnxTrBackend.OpenVino)
        {
            options.AppendExecutionProvider_OpenVINO();
        }

        return new InferenceSession(modelPath, options);
    }

    private async Task<string> ResolveModelPathAsync(string? customPath, Uri source, CancellationToken cancellationToken)
    {
        if (!string.IsNullOrWhiteSpace(customPath))
        {
            return Path.GetFullPath(customPath);
        }

        var cacheDirectory = ResolveCacheDirectory(_options.ModelCacheDirectory);
        Directory.CreateDirectory(cacheDirectory);
        var fileName = Path.GetFileName(source.LocalPath);
        var destination = Path.Combine(cacheDirectory, fileName);

        if (File.Exists(destination))
        {
            return destination;
        }

        using var response = await HttpClient.GetAsync(source, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
        await using var fileStream = File.Create(destination);
        await stream.CopyToAsync(fileStream, cancellationToken).ConfigureAwait(false);

        return destination;
    }

    private static string ResolveCacheDirectory(string? customDirectory)
    {
        if (!string.IsNullOrWhiteSpace(customDirectory))
        {
            return Path.GetFullPath(customDirectory);
        }

        var baseDirectory = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        if (string.IsNullOrWhiteSpace(baseDirectory))
        {
            baseDirectory = Directory.GetCurrentDirectory();
        }

        return Path.Combine(baseDirectory, "OnnxTr", "models");
    }

    private static (int Index, float Probability) ArgMaxWithProbability(ReadOnlySpan<float> logits)
    {
        var maxIndex = 0;
        var maxLogit = float.NegativeInfinity;
        for (var i = 0; i < logits.Length; i++)
        {
            if (logits[i] > maxLogit)
            {
                maxLogit = logits[i];
                maxIndex = i;
            }
        }

        var sum = 0f;
        for (var i = 0; i < logits.Length; i++)
        {
            sum += MathF.Exp(logits[i] - maxLogit);
        }

        var probability = MathF.Exp(logits[maxIndex] - maxLogit) / sum;
        return (maxIndex, probability);
    }

    private static float Sigmoid(float value) => 1f / (1f + MathF.Exp(-value));

    private static float Clamp01(float value) => Math.Clamp(value, 0f, 1f);

    private readonly record struct DetectionInput(DenseTensor<float> Tensor, DetectionMetadata Metadata);

    private readonly record struct DetectionMetadata(
        int OriginalWidth,
        int OriginalHeight,
        int TargetWidth,
        int TargetHeight,
        float ScaleX,
        float ScaleY,
        int OffsetX,
        int OffsetY);

    private readonly record struct NormalizedBox(float XMin, float YMin, float XMax, float YMax, float Score);

    private readonly record struct WordPrediction(string Text, float Confidence, NormalizedBox Box, float ObjectnessScore);

    private sealed record DetectionModelConfig(
        string Name,
        Uri SourceUri,
        int InputHeight,
        int InputWidth,
        IReadOnlyList<float> Mean,
        IReadOnlyList<float> Std,
        float BinThreshold,
        float BoxThreshold);

    private sealed record RecognitionModelConfig(
        string Name,
        Uri SourceUri,
        int InputHeight,
        int InputWidth,
        IReadOnlyList<float> Mean,
        IReadOnlyList<float> Std,
        string Vocabulary)
    {
        public int ClassCount => Vocabulary.Length + 1;
    }
}
