namespace OnnxTr.Sdk;

/// <summary>
/// Represents an image that should be processed by the OCR pipeline.
/// </summary>
public sealed class OnnxTrImageInput
{
    private OnnxTrImageInput(string? filePath, Stream? content, string? fileName)
    {
        if (filePath is null && content is null)
        {
            throw new ArgumentException("Either a file path or a stream must be provided.", nameof(filePath));
        }

        FilePath = filePath;
        Content = content;
        FileName = fileName;
    }

    /// <summary>
    /// Gets the absolute file path that points to the image on disk, when available.
    /// </summary>
    public string? FilePath { get; }

    /// <summary>
    /// Gets the stream that contains the raw image data, when available.
    /// </summary>
    public Stream? Content { get; }

    /// <summary>
    /// Gets the optional file name hint that will be used when materialising the stream on disk.
    /// </summary>
    public string? FileName { get; }

    /// <summary>
    /// Creates an input that references an image that already exists on disk.
    /// </summary>
    /// <param name="filePath">The path of the image to process.</param>
    /// <returns>A new <see cref="OnnxTrImageInput"/> instance.</returns>
    public static OnnxTrImageInput FromFile(string filePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath);

        return new OnnxTrImageInput(Path.GetFullPath(filePath), null, Path.GetFileName(filePath));
    }

    /// <summary>
    /// Creates an input that consumes an in-memory stream.
    /// </summary>
    /// <param name="content">The stream that contains the image data.</param>
    /// <param name="fileName">
    /// Optional name hint (including extension) that will be used when the SDK writes the stream on disk.
    /// When not provided a PNG extension is assumed.
    /// </param>
    /// <returns>A new <see cref="OnnxTrImageInput"/> instance.</returns>
    public static OnnxTrImageInput FromStream(Stream content, string? fileName = null)
    {
        ArgumentNullException.ThrowIfNull(content);

        if (!content.CanRead)
        {
            throw new ArgumentException("The provided stream must be readable.", nameof(content));
        }

        return new OnnxTrImageInput(null, content, fileName);
    }

    internal async Task<string> MaterialiseAsync(string directory, int index, CancellationToken cancellationToken)
    {
        if (FilePath is not null)
        {
            return FilePath;
        }

        ArgumentNullException.ThrowIfNull(Content);

        var extension = Path.GetExtension(FileName);
        if (string.IsNullOrEmpty(extension))
        {
            extension = ".png";
        }

        var resolvedFileName = !string.IsNullOrWhiteSpace(FileName)
            ? FileName!
            : $"page_{index:D4}{extension}";

        var targetPath = Path.Combine(directory, resolvedFileName);
        Directory.CreateDirectory(Path.GetDirectoryName(targetPath)!);

        if (Content.CanSeek)
        {
            Content.Seek(0, SeekOrigin.Begin);
        }

        await using var fileStream = File.Create(targetPath);
        await Content.CopyToAsync(fileStream, cancellationToken);

        return targetPath;
    }
}
