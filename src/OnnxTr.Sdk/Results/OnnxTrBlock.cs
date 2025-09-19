using System.Text.Json.Serialization;

namespace OnnxTr.Sdk.Results;

/// <summary>
/// Represents a block of text on a page.
/// </summary>
public sealed class OnnxTrBlock
{
    [JsonPropertyName("geometry")]
    public IReadOnlyList<IReadOnlyList<float>>? Geometry { get; init; }

    [JsonPropertyName("objectness_score")]
    public float? ObjectnessScore { get; init; }

    [JsonPropertyName("lines")]
    public List<OnnxTrLine> Lines { get; init; } = new();

    [JsonPropertyName("artefacts")]
    public List<OnnxTrArtefact> Artefacts { get; init; } = new();
}
