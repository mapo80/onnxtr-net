using System.Text.Json.Serialization;

namespace OnnxTr.Sdk.Results;

/// <summary>
/// Represents a non textual element detected by the OCR pipeline (for example logos or separators).
/// </summary>
public sealed class OnnxTrArtefact
{
    [JsonPropertyName("geometry")]
    public IReadOnlyList<IReadOnlyList<float>>? Geometry { get; init; }

    [JsonPropertyName("type")]
    public string? Type { get; init; }

    [JsonPropertyName("confidence")]
    public float Confidence { get; init; }
}
