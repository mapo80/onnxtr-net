namespace OnnxTr.Sdk;

/// <summary>
/// Provides configuration options for <see cref="OnnxTrClient"/>.
/// </summary>
public sealed class OnnxTrClientOptions
{
    /// <summary>
    /// Gets or sets the directory where downloaded ONNX models should be cached. When not specified the
    /// client stores models inside the current user's local application data folder under
    /// <c>OnnxTr/models</c>.
    /// </summary>
    public string? ModelCacheDirectory { get; set; }

    /// <summary>
    /// Gets or sets the path of a custom detection model. When specified the SDK will load this file
    /// instead of downloading the artefact associated with <see cref="OnnxTrRunOptions.DetectionModel"/>.
    /// </summary>
    public string? DetectionModelPath { get; set; }

    /// <summary>
    /// Gets or sets the path of a custom recognition model. When specified the SDK will load this file
    /// instead of downloading the artefact associated with <see cref="OnnxTrRunOptions.RecognitionModel"/>.
    /// </summary>
    public string? RecognitionModelPath { get; set; }
}
