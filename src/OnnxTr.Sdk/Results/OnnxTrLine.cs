using System.Text.Json.Serialization;

namespace OnnxTr.Sdk.Results;

/// <summary>
/// Represents a line of text.
/// </summary>
public sealed class OnnxTrLine
{
    [JsonPropertyName("geometry")]
    public IReadOnlyList<IReadOnlyList<float>>? Geometry { get; init; }

    [JsonPropertyName("objectness_score")]
    public float? ObjectnessScore { get; init; }

    [JsonPropertyName("words")]
    public List<OnnxTrWord> Words { get; init; } = new();
}
