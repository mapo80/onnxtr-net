using System.Text.Json.Serialization;

namespace OnnxTr.Sdk.Results;

/// <summary>
/// Represents an OCR word.
/// </summary>
public sealed class OnnxTrWord
{
    [JsonPropertyName("value")]
    public string Value { get; init; } = string.Empty;

    [JsonPropertyName("confidence")]
    public float Confidence { get; init; }

    [JsonPropertyName("geometry")]
    public IReadOnlyList<IReadOnlyList<float>>? Geometry { get; init; }

    [JsonPropertyName("objectness_score")]
    public float? ObjectnessScore { get; init; }

    [JsonPropertyName("crop_orientation")]
    public OnnxTrOrientation? CropOrientation { get; init; }
}
