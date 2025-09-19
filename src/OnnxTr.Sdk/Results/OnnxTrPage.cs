using System.Text.Json.Serialization;

namespace OnnxTr.Sdk.Results;

/// <summary>
/// Represents a single page inside an OCR document.
/// </summary>
public sealed class OnnxTrPage
{
    /// <summary>
    /// Gets or sets the index of the page within the input document.
    /// </summary>
    [JsonPropertyName("page_idx")]
    public int PageIndex { get; init; }

    /// <summary>
    /// Gets or sets the original page dimensions in pixels, encoded as (height, width).
    /// </summary>
    [JsonPropertyName("dimensions")]
    public IReadOnlyList<float>? Dimensions { get; init; }

    /// <summary>
    /// Gets or sets the optional orientation prediction.
    /// </summary>
    [JsonPropertyName("orientation")]
    public OnnxTrOrientation? Orientation { get; init; }

    /// <summary>
    /// Gets or sets the optional language prediction.
    /// </summary>
    [JsonPropertyName("language")]
    public OnnxTrLanguage? Language { get; init; }

    /// <summary>
    /// Gets or sets the blocks that compose the page.
    /// </summary>
    [JsonPropertyName("blocks")]
    public List<OnnxTrBlock> Blocks { get; init; } = new();
}
