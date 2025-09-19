using System.Text.Json.Serialization;

namespace OnnxTr.Sdk.Results;

/// <summary>
/// Represents the result of an OCR invocation.
/// </summary>
public sealed class OnnxTrResult
{
    /// <summary>
    /// Gets the backend that was used to produce the result.
    /// </summary>
    [JsonPropertyName("backend")]
    public string Backend { get; init; } = string.Empty;

    /// <summary>
    /// Gets the document that was extracted from the images.
    /// </summary>
    [JsonPropertyName("document")]
    public OnnxTrDocument Document { get; init; } = new();

    /// <summary>
    /// Gets the text rendered from the document.
    /// </summary>
    [JsonPropertyName("text")]
    public string Text { get; init; } = string.Empty;

    /// <summary>
    /// Gets the individual OCR pages contained in the document.
    /// </summary>
    [JsonIgnore]
    public IReadOnlyList<OnnxTrPage> Pages => Document.Pages;
}
