# OnnxTR .NET SDK

Questa soluzione fornisce una libreria .NET (`OnnxTr.Sdk`) che esegue la pipeline OCR di
[OnnxTR](https://pypi.org/project/onnxtr/) direttamente da codice gestito utilizzando
[Microsoft.ML.OnnxRuntime](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime). Non è richiesto
alcun interprete Python: i modelli di detection e recognition vengono scaricati in formato ONNX e
processati tramite Onnx Runtime.

La libreria supporta sia il backend CPU predefinito, sia il provider
[OpenVINO](https://www.nuget.org/packages/Intel.ML.OnnxRuntime.OpenVino) quando disponibile sul
sistema.

## Struttura del progetto

- `OnnxTrNet.sln`: soluzione principale.
- `src/OnnxTr.Sdk`: libreria .NET che espone l'API di alto livello.
  - `OnnxTrClient`: classe principale per eseguire inferenza OCR su immagini.
  - `OnnxTrClientOptions`: opzioni per configurare cache e percorsi dei modelli.
  - `OnnxTrRunOptions`: opzioni runtime per scegliere le architetture dei modelli.
  - `Results/*`: modelli tipizzati che descrivono il risultato dell'OCR.

## Installazione

Il progetto utilizza i seguenti pacchetti NuGet:

- `Microsoft.ML.OnnxRuntime`
- `Intel.ML.OnnxRuntime.OpenVino` (opzionale, per eseguire su backend OpenVINO)
- `SkiaSharp`

Per includere la libreria in un'applicazione è sufficiente referenziare il progetto `OnnxTr.Sdk` o
pubblicarlo come pacchetto NuGet interno.

## Utilizzo

```csharp
using OnnxTr.Sdk;
using OnnxTr.Sdk.Results;

var client = new OnnxTrClient(OnnxTrBackend.Cpu);

var images = new[]
{
    OnnxTrImageInput.FromFile("invoice-page1.jpg"),
    OnnxTrImageInput.FromFile("invoice-page2.jpg"),
};

OnnxTrResult result = await client.PredictAsync(images);

Console.WriteLine(result.Text);
foreach (var page in result.Pages)
{
    Console.WriteLine($"Page {page.PageIndex} -> {page.Blocks.Count} blocks");
}
```

Per utilizzare il backend OpenVINO è sufficiente istanziare il client con
`OnnxTrBackend.OpenVino`. Se il runtime non trova le librerie OpenVINO necessarie, lancerà un'
eccezione durante la creazione della sessione ONNX.

### Opzioni client

`OnnxTrClientOptions` consente di controllare dove vengono memorizzati i modelli ONNX e di
specificare file personalizzati:

```csharp
var options = new OnnxTrClientOptions
{
    ModelCacheDirectory = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "MyApp", "onnx"),
    DetectionModelPath = "./models/custom-detection.onnx",
    RecognitionModelPath = "./models/custom-recognition.onnx",
};

var client = new OnnxTrClient(OnnxTrBackend.Cpu, options);
```

Quando non vengono forniti percorsi personalizzati il client scarica automaticamente i modelli
ufficiali da GitHub e li memorizza nella cartella `OnnxTr/models` all'interno della directory dati
locale dell'utente.

### Opzioni runtime

`OnnxTrRunOptions` permette di selezionare tramite enum i modelli di detection e recognition supportati
e di attivare (o meno) funzionalità aggiuntive. La pipeline nativa attualmente non supporta
`DetectOrientation`, `DetectLanguage` e `StraightenPages` e genererà un'eccezione se impostati a `true`.

```csharp
var runOptions = new OnnxTrRunOptions
{
    DetectionModel = OnnxTrDetectionModel.FastSmall,
    RecognitionModel = OnnxTrRecognitionModel.CrnnMobilenetV3Small,
};

var client = new OnnxTrClient(OnnxTrBackend.Cpu, runOptions: runOptions);
```

### Guida pratica all'uso dei modelli

Per utilizzare i modelli ONNX/OpenVINO pubblicati con questo progetto in un'applicazione .NET è
possibile seguire il flusso operativo seguente:

1. **Scarica gli artefatti** dalla release GitHub del repository (`.onnx` oppure coppie `.xml`/`.bin`).
2. **Organizza i file** in una directory accessibile dall'applicazione, ad esempio
   `./models/onnxtr` per i modelli ONNX oppure `./models/openvino` per le controparti IR.
3. **Configura `OnnxTrClientOptions`** indicando il percorso della cartella e dei file specifici che
   vuoi utilizzare. È possibile specificare solo i file che differiscono da quelli predefiniti.
4. **Scegli l'architettura desiderata** tramite `OnnxTrRunOptions` e gli enum `OnnxTrDetectionModel`
   / `OnnxTrRecognitionModel` così da allineare il codice ai modelli scaricati.
5. **Istanzia `OnnxTrClient`** passando backend, opzioni e run options; il client riutilizzerà i
   modelli già presenti senza tentare di riscaricarli.

Esempio pratico che utilizza modelli ONNX copiati localmente:

```csharp
var options = new OnnxTrClientOptions
{
    ModelCacheDirectory = Path.Combine(AppContext.BaseDirectory, "models"),
    DetectionModelPath = Path.Combine(AppContext.BaseDirectory, "models", "rep_fast_small-10428b70.onnx"),
    RecognitionModelPath = Path.Combine(AppContext.BaseDirectory, "models", "crnn_mobilenet_v3_small-bded4d49.onnx"),
};

var runOptions = new OnnxTrRunOptions
{
    DetectionModel = OnnxTrDetectionModel.FastSmall,
    RecognitionModel = OnnxTrRecognitionModel.CrnnMobilenetV3Small,
};

await using var client = new OnnxTrClient(OnnxTrBackend.Cpu, options, runOptions);
var result = await client.PredictAsync(new[] { OnnxTrImageInput.FromFile("./input.png") });
```

Quando si desidera utilizzare OpenVINO è sufficiente convertire i modelli ONNX nel formato IR o
scaricare direttamente gli artefatti `.xml`/`.bin` dalla release, quindi impostare `DetectionModelPath`
e `RecognitionModelPath` verso i file `.xml` corrispondenti e istanziare il client con
`OnnxTrBackend.OpenVino`.

## Requisiti

- .NET 8.0 o superiore.
- Accesso a Internet al primo avvio per scaricare i modelli ONNX (a meno che non si utilizzi
  `DetectionModelPath`/`RecognitionModelPath`).
- Per l'esecuzione con backend OpenVINO potrebbero essere necessarie le librerie native di OpenVINO
  corrispondenti al sistema operativo.

## Modelli pubblicati

Questo progetto fa riferimento ai modelli ufficiali messi a disposizione dal repository
[`OnnxTR`](https://github.com/felixdittrich92/OnnxTR/releases). Per facilitare la distribuzione dello
SDK è consigliabile creare una _release_ GitHub del presente repository che includa i seguenti
artefatti:

### Formato ONNX

- `rep_fast_tiny-28867779.onnx` &rarr; modello di **text detection** più compatto, pensato per dispositivi a
  bassa memoria dove conta la latenza più che la precisione.
- `rep_fast_small-10428b70.onnx` &rarr; variante intermedia della rete di **text detection** che bilancia
  velocità e accuratezza per carichi server leggeri.
- `rep_fast_base-1b89ebf9.onnx` &rarr; modello di **text detection** più pesante, indicato quando serve la
  massima precisione su documenti complessi.
- `crnn_mobilenet_v3_small-bded4d49.onnx` &rarr; modello di **text recognition** basato su MobileNetV3 small,
  ottimizzato per ambienti CPU con vincoli di risorse.
- `crnn_mobilenet_v3_large-d42e8185.onnx` &rarr; variante **text recognition** MobileNetV3 large che aumenta la
  qualità dell'estrazione del testo mantenendo una buona efficienza.
- `crnn_vgg16_bn-743599aa.onnx` &rarr; modello di **text recognition** più accurato (baseline VGG16 con batch
  normalization) consigliato per scenari dove la precisione è prioritaria rispetto alla velocità.

La presenza di tre modelli di detection e tre di recognition permette di scegliere la combinazione più
adatta alle risorse hardware e agli obiettivi di qualità dell'OCR.

### Formato OpenVINO (IR)

- `rep_fast_tiny.xml` / `rep_fast_tiny.bin` &rarr; rappresentano la conversione in **Intermediate
  Representation** del modello `rep_fast_tiny`, necessaria a OpenVINO per eseguire la detection su CPU e VPU.
- `rep_fast_small.xml` / `rep_fast_small.bin` &rarr; coppia IR della variante intermedia di **text detection**.
- `rep_fast_base.xml` / `rep_fast_base.bin` &rarr; coppia IR del modello di **text detection** più accurato.
- `crnn_mobilenet_v3_small.xml` / `crnn_mobilenet_v3_small.bin` &rarr; conversione IR del modello MobileNetV3
  small di **text recognition**.
- `crnn_mobilenet_v3_large.xml` / `crnn_mobilenet_v3_large.bin` &rarr; conversione IR della variante MobileNetV3
  large di **text recognition**.
- `crnn_vgg16_bn.xml` / `crnn_vgg16_bn.bin` &rarr; conversione IR del modello VGG16 batch-normalized di **text
  recognition**.

Per OpenVINO ogni rete è composta da due file: `.xml` descrive la topologia e i layer, mentre `.bin`
contiene i pesi. Entrambi sono indispensabili per l'esecuzione e vengono generati dal Model Optimizer a
partire dagli ONNX corrispondenti.

Al momento la pipeline di build non può pubblicare automaticamente tali file; è quindi necessario
caricarli manualmente nella sezione "Assets" della release una volta generati (ad esempio tramite
`mo` di OpenVINO partendo dai modelli ONNX elencati sopra).

## Build

```bash
dotnet build
```

Il comando precedente compila l'SDK; non è necessario alcun ambiente Python.
