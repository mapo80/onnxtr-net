namespace OnnxTr.Sdk;

/// <summary>
/// Exposes high level options that control how the OnnxTR OCR pipeline is executed.
/// </summary>
public sealed class OnnxTrRunOptions
{
    /// <summary>
    /// Gets or sets the detection model that should be used by the OCR pipeline. The default value is
    /// <see cref="OnnxTrDetectionModel.FastBase"/>, which provides the best accuracy among the published
    /// architectures.
    /// </summary>
    public OnnxTrDetectionModel DetectionModel { get; set; } = OnnxTrDetectionModel.FastBase;

    /// <summary>
    /// Gets or sets the recognition model that should be used by the OCR pipeline. The default value is
    /// <see cref="OnnxTrRecognitionModel.CrnnVgg16Bn"/>, which offers the highest quality predictions.
    /// </summary>
    public OnnxTrRecognitionModel RecognitionModel { get; set; } = OnnxTrRecognitionModel.CrnnVgg16Bn;

    /// <summary>
    /// Gets or sets a value indicating whether the pipeline should attempt to detect the orientation of
    /// each page. The current native .NET pipeline does not expose orientation detection and will throw
    /// if this flag is enabled.
    /// </summary>
    public bool DetectOrientation { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether the pipeline should attempt to detect the language of the
    /// input document. The current native .NET pipeline does not expose language detection and will throw
    /// if this flag is enabled.
    /// </summary>
    public bool DetectLanguage { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether the pages should be straightened before the recognition
    /// step. This maps to the <c>straighten_pages</c> flag of the OnnxTR predictor and is not supported by
    /// the native .NET pipeline.
    /// </summary>
    public bool StraightenPages { get; set; }
}
