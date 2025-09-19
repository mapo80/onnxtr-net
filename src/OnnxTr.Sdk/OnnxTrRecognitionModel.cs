namespace OnnxTr.Sdk;

/// <summary>
/// Enumerates the available text recognition ONNX models shipped with OnnxTR.
/// </summary>
public enum OnnxTrRecognitionModel
{
    /// <summary>
    /// Uses the CRNN MobileNetV3 small recognition model optimised for CPU inference.
    /// </summary>
    CrnnMobilenetV3Small,

    /// <summary>
    /// Uses the CRNN MobileNetV3 large recognition model for improved quality while maintaining efficiency.
    /// </summary>
    CrnnMobilenetV3Large,

    /// <summary>
    /// Uses the CRNN VGG16 batch-normalised recognition model that offers the highest accuracy.
    /// </summary>
    CrnnVgg16Bn,
}
