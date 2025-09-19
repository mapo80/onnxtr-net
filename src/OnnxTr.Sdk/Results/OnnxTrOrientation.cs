using System.Text.Json.Serialization;

namespace OnnxTr.Sdk.Results;

/// <summary>
/// Describes the orientation prediction for a page or a crop.
/// </summary>
public sealed class OnnxTrOrientation
{
    [JsonPropertyName("value")]
    public float Value { get; init; }

    [JsonPropertyName("confidence")]
    public float Confidence { get; init; }
}
