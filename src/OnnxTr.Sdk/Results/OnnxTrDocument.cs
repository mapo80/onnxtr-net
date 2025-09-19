using System.Text.Json.Serialization;

namespace OnnxTr.Sdk.Results;

/// <summary>
/// Describes the full OCR document as exported by OnnxTR.
/// </summary>
public sealed class OnnxTrDocument
{
    /// <summary>
    /// Gets or sets the collection of pages.
    /// </summary>
    [JsonPropertyName("pages")]
    public List<OnnxTrPage> Pages { get; init; } = new();
}
