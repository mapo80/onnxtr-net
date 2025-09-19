namespace OnnxTr.Sdk;

/// <summary>
/// Specifies the execution backend that should be used to run the OnnxTR OCR models.
/// </summary>
public enum OnnxTrBackend
{
    /// <summary>
    /// Uses the default ONNX Runtime CPU execution provider.
    /// </summary>
    Cpu,

    /// <summary>
    /// Uses the OpenVINO execution provider (requires the <c>onnxtr[openvino]</c> python package).
    /// </summary>
    OpenVino,
}
