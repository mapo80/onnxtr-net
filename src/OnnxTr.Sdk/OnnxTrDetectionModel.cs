namespace OnnxTr.Sdk;

/// <summary>
/// Enumerates the available text detection ONNX models shipped with OnnxTR.
/// </summary>
public enum OnnxTrDetectionModel
{
    /// <summary>
    /// Uses the lightest detection model optimised for latency on constrained hardware.
    /// </summary>
    FastTiny,

    /// <summary>
    /// Uses the mid-sized detection model that balances accuracy and speed.
    /// </summary>
    FastSmall,

    /// <summary>
    /// Uses the most accurate detection model, recommended when maximum quality is required.
    /// </summary>
    FastBase,
}
