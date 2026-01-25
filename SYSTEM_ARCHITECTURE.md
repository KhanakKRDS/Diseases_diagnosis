# Disease Diagnosis System - Complete Architecture & Flow

---

## 1. SYSTEM ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────────┐
│                       USER BROWSER                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         Frontend: website.html (UI)                      │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  1. File Input - User selects X-ray image         │  │  │
│  │  │  2. Preview - Image shown to user                 │  │  │
│  │  │  3. Predict Button - Triggers upload & inference  │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ▼
                      [POST HTTP Request]
                   (FormData with image file)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Backend: FastAPI Server (Port 8000)                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  /predict/ Endpoint                                      │  │
│  │  ├─ Receive uploaded image bytes                         │  │
│  │  ├─ Preprocess: Resize, Grayscale, Normalize            │  │
│  │  ├─ Run CNN Model                                        │  │
│  │  └─ Return JSON: {"prediction": "Normal/Pneumonia/TB"}  │  │
│  │                                                           │  │
│  │  Model Storage: xray_cnn_model.pth (trained weights)    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ▼
                      [JSON HTTP Response]
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       BROWSER (again)                           │
│         Display Result: "Prediction: [Class Name]"             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. DETAILED PREDICTION FLOW (Step-by-Step)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PREDICTION WORKFLOW SEQUENCE                           │
└─────────────────────────────────────────────────────────────────────────────┘

STEP 1: USER UPLOADS IMAGE (Frontend)
┌─────────────────────────────────────────┐
│ User selects X-ray file from computer   │
│ File: chest_xray.jpg (e.g., 500 KB)     │
│                                         │
│ JavaScript preview handler:             │
│  - Reads file from input                │
│  - Displays preview in <img> tag        │
└─────────────────────────────────────────┘
                    ▼

STEP 2: USER CLICKS PREDICT BUTTON (Frontend)
┌─────────────────────────────────────────┐
│ JavaScript Event: predictButton.click()  │
│                                         │
│ Actions:                                │
│ 1. Get file from input                  │
│ 2. Create FormData object               │
│ 3. Append file to FormData              │
│ 4. Show "Predicting..." message         │
└─────────────────────────────────────────┘
                    ▼

STEP 3: SEND HTTP POST REQUEST (Frontend)
┌─────────────────────────────────────────────────────────┐
│ fetch('http://127.0.0.1:8000/predict/', {              │
│   method: 'POST',                                       │
│   body: formData    ← Contains image file               │
│ })                                                      │
│                                                         │
│ Network: Browser → FastAPI Server (localhost:8000)     │
└─────────────────────────────────────────────────────────┘
                    ▼

STEP 4: BACKEND RECEIVES REQUEST (Backend)
┌──────────────────────────────────────────────────────────┐
│ FastAPI: @app.post("/predict/")                          │
│ Function: async def predict_disease(file: UploadFile)    │
│                                                          │
│ Action: await file.read()  ← Get image bytes            │
│ Result: bytes object (raw image data)                   │
└──────────────────────────────────────────────────────────┘
                    ▼

STEP 5: PREPROCESS IMAGE (Backend)
┌──────────────────────────────────────────────────────────┐
│ Input: Raw image bytes                                   │
│                                                          │
│ 5a. Convert to PIL Image:                               │
│     Image.open(io.BytesIO(image_data)).convert("RGB")   │
│     Result: PIL Image (variable size)                   │
│                                                          │
│ 5b. Apply Transform Pipeline:                           │
│     • Resize → (128, 128) pixels                        │
│     • Grayscale → 3-channel (for compatibility)         │
│     • ToTensor → PyTorch tensor, values [0, 1]          │
│     Result: Tensor shape (3, 128, 128)                  │
│                                                          │
│ 5c. Add Batch Dimension:                                │
│     unsqueeze(0)  ← Add dimension at position 0         │
│     Result: Tensor shape (1, 3, 128, 128)               │
│     ^ batch  ^ channels ^ height ^ width                │
│                                                          │
│ 5d. Move to Device:                                      │
│     .to(device)  ← GPU if available, else CPU           │
│     Result: Ready for model inference                   │
└──────────────────────────────────────────────────────────┘
                    ▼

STEP 6: LOAD MODEL & WEIGHTS (Backend - happens once at startup)
┌──────────────────────────────────────────────────────────┐
│ Model Architecture Loaded:                               │
│ DiseaseDiagnosisModel(num_classes=3)                     │
│                                                          │
│ Weights Loaded from File:                               │
│ model.load_state_dict(torch.load(                        │
│   'xray_cnn_model.pth',                                  │
│   map_location=device                                    │
│ ))                                                       │
│                                                          │
│ Model State:                                             │
│ model.to(device)  ← Move to GPU/CPU                      │
│ model.eval()      ← Set to evaluation mode               │
│                   (disable dropout, set BN to eval)      │
└──────────────────────────────────────────────────────────┘
                    ▼

STEP 7: RUN MODEL INFERENCE (Backend - THE KEY STEP)
┌──────────────────────────────────────────────────────────┐
│ with torch.no_grad():  ← Don't track gradients           │
│   outputs = model(input_tensor)                          │
│                                                          │
│ Input Shape:  (1, 3, 128, 128)                           │
│ Output Shape: (1, 3)  where 3 = num_classes             │
│ Output Example: [[2.5, -1.3, 0.8]]                       │
│                 ↑Normal, Pneumonia, Tuberculosis (logits)│
│                                                          │
│ The model's CNN processes the image through:             │
│   1. Feature extraction layers (Conv2d)                  │
│   2. Classification layers (Linear)                      │
│   3. Produces a score for each disease class             │
└──────────────────────────────────────────────────────────┘
                    ▼

STEP 8: POST-PROCESS OUTPUT (Backend)
┌──────────────────────────────────────────────────────────┐
│ outputs = [[2.5, -1.3, 0.8]]  ← Raw model output        │
│                                                          │
│ _, predicted = torch.max(outputs, 1)                    │
│   Find max value in dimension 1:                         │
│   Max: 2.5 at index 0 ← This is class 0                 │
│   Result: predicted = [0]                               │
│                                                          │
│ predicted_class = class_names[predicted.item()]          │
│   class_names = ['Normal', 'Pneumonia', 'Tuberculosis'] │
│   Result: 'Normal'                                       │
│                                                          │
│ Create JSON Response:                                    │
│ {"prediction": "Normal"}                                 │
└──────────────────────────────────────────────────────────┘
                    ▼

STEP 9: SEND JSON RESPONSE (Backend)
┌──────────────────────────────────────────────────────────┐
│ FastAPI returns JSON:                                    │
│ {"prediction": "Normal"}                                 │
│                                                          │
│ Network: FastAPI Server → Browser                        │
└──────────────────────────────────────────────────────────┘
                    ▼

STEP 10: DISPLAY RESULT (Frontend)
┌──────────────────────────────────────────────────────────┐
│ await fetch(...).json()  ← Parse response                │
│ const data = response object                             │
│                                                          │
│ resultText.textContent = "Prediction: " + data.prediction│
│ Result shown in browser: "Prediction: Normal"            │
│                                                          │
│ Also display note:                                       │
│ "Note: This is diagnosed using an AI. This is not       │
│  made to replace Doctors."                              │
└──────────────────────────────────────────────────────────┘
```

---

## 3. CNN ARCHITECTURE IN DETAIL

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CNN LAYER STRUCTURE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT: X-ray Image
┌─────────────────────┐
│  Preprocessed       │
│  Tensor             │
│  Shape:             │
│  (1, 3, 128, 128)   │
│  batch=1            │
│  channels=3 (RGB)   │
│  height=128         │
│  width=128          │
└─────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│ FEATURE EXTRACTION BLOCK (self.features)                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│ ┌─ LAYER 1: Conv2d(3 → 32, kernel=3×3) ─┐                          │
│ │  Input:  (1, 3, 128, 128)              │                          │
│ │  Learns: 32 filters (3×3 kernels)      │                          │
│ │  Output: (1, 32, 128, 128)             │ Detects edges, textures  │
│ │  Operation:                            │                          │
│ │    - Slides 3×3 window across image    │                          │
│ │    - Computes dot product with weights │                          │
│ │    - Produces 32 feature maps          │                          │
│ └────────────────────────────────────────┘                          │
│          │                                                           │
│          ▼                                                           │
│ ┌─ ReLU Activation ─┐                                               │
│ │ max(0, x)         │ Introduces non-linearity                      │
│ │ Negative → 0      │                                               │
│ │ Positive → keep   │                                               │
│ │ Shape: (1, 32, 128, 128) [unchanged]                             │
│ └───────────────────┘                                               │
│          │                                                           │
│          ▼                                                           │
│ ┌─ MaxPool2d(2×2) ─┐                                                │
│ │ Takes max of      │ Reduces spatial size                          │
│ │ 2×2 windows       │ Keeps important features                      │
│ │ Stride: 2         │ Reduces computation                           │
│ │ Shape: (1, 32, 64, 64)                                            │
│ │    128÷2 = 64     │                                               │
│ └───────────────────┘                                               │
│          │                                                           │
│          ▼                                                           │
│ ┌─ LAYER 2: Conv2d(32 → 64, kernel=3×3) ─┐                         │
│ │  Input:  (1, 32, 64, 64)                │                        │
│ │  Learns: 64 filters (3×3 kernels)       │ Detects textures,      │
│ │  Output: (1, 64, 64, 64)                │ patterns               │
│ │  Weights: 32×3×3×64 parameters          │                        │
│ └─────────────────────────────────────────┘                        │
│          │                                                           │
│          ▼                                                           │
│ ┌─ ReLU Activation ─┐                                               │
│ │ Shape: (1, 64, 64, 64)                                            │
│ └───────────────────┘                                               │
│          │                                                           │
│          ▼                                                           │
│ ┌─ MaxPool2d(2×2) ─┐                                                │
│ │ Shape: (1, 64, 32, 32)                                            │
│ │    64÷2 = 32     │                                                │
│ └───────────────────┘                                               │
│          │                                                           │
│          ▼                                                           │
│ ┌─ LAYER 3: Conv2d(64 → 128, kernel=3×3) ─┐                        │
│ │  Input:  (1, 64, 32, 32)                 │                       │
│ │  Learns: 128 filters (3×3 kernels)       │ High-level features   │
│ │  Output: (1, 128, 32, 32)                │ (disease patterns)     │
│ │  Weights: 64×3×3×128 parameters          │                       │
│ └──────────────────────────────────────────┘                       │
│          │                                                           │
│          ▼                                                           │
│ ┌─ ReLU Activation ─┐                                               │
│ │ Shape: (1, 128, 32, 32)                                           │
│ └───────────────────┘                                               │
│          │                                                           │
│          ▼                                                           │
│ ┌─ MaxPool2d(2×2) ─┐                                                │
│ │ Shape: (1, 128, 16, 16)                                           │
│ │    32÷2 = 16     │                                                │
│ └───────────────────┘                                               │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
         │
         ▼ Output of Features: (1, 128, 16, 16)
         │  = 1 batch × 128 channels × 16 pixels × 16 pixels
         │  = Total values: 1 × 128 × 256 = 32,768 numbers
         │
┌──────────────────────────────────────────────────────────────────────┐
│ FLATTEN                                                               │
├──────────────────────────────────────────────────────────────────────┤
│  x.view(x.size(0), -1)                                              │
│  Reshape (1, 128, 16, 16) → (1, 32768)                             │
│  Convert 2D feature map into 1D vector                              │
│  [[[...]  [...]]]  →  [... ... ... ... (32768 values) ...]         │
└──────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│ CLASSIFICATION BLOCK (self.classifier)                               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│ ┌─ Dropout(50%) ─┐                                                   │
│ │ During training:                  │ Prevents overfitting           │
│ │   - Randomly disable ~50% neurons │                               │
│ │ During eval:                      │                               │
│ │   - Keep all neurons (disabled)   │                               │
│ │ Shape: (1, 32768) [unchanged]     │                               │
│ └─────────────────┘                                                  │
│          │                                                            │
│          ▼                                                            │
│ ┌─ Linear(32768 → 512) ─┐                                            │
│ │ Input:  (1, 32768)     │                                           │
│ │ Weights: 32768×512 = 16.7M parameters                            │
│ │ Biases: 512 parameters                                            │
│ │ Output: (1, 512)       │ Combines features into                   │
│ │ Formula: y = Wx + b    │ 512 abstract representations             │
│ │ W shape: (32768, 512)  │                                          │
│ │ b shape: (512,)        │                                          │
│ └────────────────────────┘                                           │
│          │                                                            │
│          ▼                                                            │
│ ┌─ ReLU Activation ─┐                                                │
│ │ Shape: (1, 512)                                                    │
│ │ max(0, x)                                                          │
│ └───────────────────┘                                                │
│          │                                                            │
│          ▼                                                            │
│ ┌─ Dropout(50%) ─┐                                                   │
│ │ Shape: (1, 512)                                                    │
│ └─────────────────┘                                                  │
│          │                                                            │
│          ▼                                                            │
│ ┌─ Linear(512 → 3) ─┐                                                │
│ │ Input:  (1, 512)               │                                   │
│ │ Weights: 512×3 = 1,536 parameters                                 │
│ │ Biases: 3 parameters            │                                  │
│ │ Output: (1, 3)                  │ Final disease predictions         │
│ │ Formula: y = Wx + b             │                                  │
│ │ Classes:                        │                                  │
│ │  - Output[0] = Normal score     │                                  │
│ │  - Output[1] = Pneumonia score  │                                  │
│ │  - Output[2] = Tuberculosis score                                  │
│ │                                 │                                  │
│ │ Example Output: [2.5, -1.3, 0.8] (logits, not probabilities)      │
│ └─────────────────────────────────┘                                  │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
         │
         ▼
OUTPUT: (1, 3)  = Logits for 3 classes
        ↓
ARGMAX: Index of max value (e.g., 0 → "Normal")
        ↓
CLASS NAME: Map index to class_names list
        ↓
RESULT: {"prediction": "Normal"}
```

---

## 4. WEIGHTS & PARAMETERS BREAKDOWN

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LEARNABLE PARAMETERS (WEIGHTS)                           │
└─────────────────────────────────────────────────────────────────────────────┘

CONVOLUTIONAL LAYERS (Feature Extraction):
├─ Conv2d(3 → 32, 3×3):
│  ├─ Kernel weights:    3×3×3×32  =        864 parameters
│  └─ Biases:                             +  32 parameters
│                                    Total: 896
│  Purpose: Learn to detect basic patterns (edges, corners, textures)
│
├─ Conv2d(32 → 64, 3×3):
│  ├─ Kernel weights:    3×3×32×64 =     18,432 parameters
│  └─ Biases:                             +  64 parameters
│                                    Total: 18,496
│  Purpose: Learn mid-level patterns (shapes, objects)
│
└─ Conv2d(64 → 128, 3×3):
   ├─ Kernel weights:    3×3×64×128 =    73,728 parameters
   └─ Biases:                             + 128 parameters
                                    Total: 73,856
   Purpose: Learn disease-specific patterns (disease signatures)

FULLY CONNECTED LAYERS (Classification):
├─ Linear(32768 → 512):
│  ├─ Weights:          32768×512 = 16,777,216 parameters
│  └─ Biases:                      +       512 parameters
│                               Total: 16,777,728
│  Purpose: Combine features into hidden representations
│
└─ Linear(512 → 3):
   ├─ Weights:          512×3 =          1,536 parameters
   └─ Biases:                        +      3 parameters
                                    Total: 1,539
   Purpose: Output disease predictions


TOTAL PARAMETERS: ~16,872,515 weights to learn!

PARAMETER STORAGE:
├─ Each parameter: 32-bit float = 4 bytes
├─ Total size: 16,872,515 × 4 ≈ 67.5 MB
├─ File: xray_cnn_model.pth (~50-60 MB with compression)
└─ When loaded to GPU: Takes VRAM proportional to model size + batch size
```

---

## 5. TRAINING VS INFERENCE DIFFERENCE

```
┌─────────────────────────────────────────────────────────────────────┐
│                  TRAINING MODE                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ model.train()  ← Enable training-specific behaviors                │
│                                                                     │
│ Forward Pass:                                                       │
│   1. Input image → Conv layers → Feature extraction                │
│   2. Features → Linear layers → Predictions                        │
│   3. Dropout: RANDOMLY disable ~50% of neurons each pass           │
│      (prevents overfitting, forces robustness)                     │
│                                                                     │
│ Loss Computation:                                                   │
│   4. outputs = model(images)                                        │
│   5. loss = criterion(outputs, labels)  ← Compare to ground truth  │
│      CrossEntropyLoss: measures how wrong predictions are          │
│                                                                     │
│ Backward Pass (Backpropagation):                                   │
│   6. loss.backward()                                                │
│      - Compute gradients: How much each weight contributed to loss │
│      - Store gradients in weight.grad tensors                      │
│                                                                     │
│ Parameter Update:                                                   │
│   7. optimizer.step()  ← Update weights using gradients            │
│      Adam optimizer: w_new = w_old - learning_rate × gradient      │
│      Weights change slightly to reduce loss                        │
│                                                                     │
│ Repeat ~1000 times per epoch with different images                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│               INFERENCE MODE (PREDICTION)                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ model.eval()  ← Disable training-specific behaviors                │
│                                                                     │
│ Forward Pass:                                                       │
│   1. Input image → Conv layers → Feature extraction                │
│   2. Features → Linear layers → Predictions                        │
│   3. Dropout: DISABLED (keep all neurons, use average)             │
│      (makes consistent predictions)                                │
│                                                                     │
│ No Loss Computation:                                                │
│   - No ground truth needed                                          │
│   - Just want the prediction, not training signal                  │
│                                                                     │
│ No Backward Pass:                                                   │
│   - with torch.no_grad():                                           │
│   - Don't compute gradients (saves memory & computation)            │
│   - Weights are NOT updated                                        │
│                                                                     │
│ Post-Process Prediction:                                            │
│   2. _, predicted = torch.max(outputs, 1)                          │
│      Find the class with highest score                             │
│   3. predicted_class = class_names[predicted.item()]               │
│      Convert index to disease name                                 │
│                                                                     │
│ Return Result:                                                      │
│   4. {"prediction": predicted_class}                                │
│                                                                     │
│ This is FAST: no backprop, no gradient tracking                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. DATA FLOW WITH SHAPES

```
INPUT: User uploads chest_xray.jpg (arbitrary size, e.g., 800×600)
│
▼
Image.open() + convert("RGB")
Shape: (H, W, 3) where H, W are original dimensions
│
▼
transforms.Resize((128, 128))
Shape: (128, 128, 3)  ← Resized to fixed size
│
▼
transforms.Grayscale(num_output_channels=3)
Shape: (3, 128, 128)  ← Converted to 3-channel (grayscale duplicated)
│
▼
transforms.ToTensor()
Shape: (3, 128, 128), values in range [0, 1]  ← PyTorch tensor
│
▼
unsqueeze(0)  ← Add batch dimension
Shape: (1, 3, 128, 128)  ← Batch size 1
│
▼
Conv2d(3→32): stride=1, padding=1
Shape: (1, 32, 128, 128)  ← Same spatial size, 32 filters
│
▼
ReLU
Shape: (1, 32, 128, 128)
│
▼
MaxPool2d(2×2)
Shape: (1, 32, 64, 64)  ← Spatial dimensions halved
│
▼
Conv2d(32→64): stride=1, padding=1
Shape: (1, 64, 64, 64)
│
▼
ReLU
Shape: (1, 64, 64, 64)
│
▼
MaxPool2d(2×2)
Shape: (1, 64, 32, 32)  ← Halved again
│
▼
Conv2d(64→128): stride=1, padding=1
Shape: (1, 128, 32, 32)
│
▼
ReLU
Shape: (1, 128, 32, 32)
│
▼
MaxPool2d(2×2)
Shape: (1, 128, 16, 16)  ← Halved third time: 128 ÷ 2 ÷ 2 ÷ 2 = 16
│
▼
view(1, -1)  ← Flatten
Shape: (1, 32768)  ← 1 × (128 × 16 × 16) = 1 × 32768
│
▼
Linear(32768→512)
Shape: (1, 512)
│
▼
ReLU
Shape: (1, 512)
│
▼
Linear(512→3)
Shape: (1, 3)  ← 3 class scores [Normal, Pneumonia, TB]
│
▼
torch.max(1)
Result: (indices, values) → Index 0, 1, or 2
│
▼
OUTPUT: {"prediction": "Normal"}
```

---

## 7. WEIGHT PERSISTENCE (.pth FILE)

```
┌────────────────────────────────────────┐
│        TRAINING (model.py)             │
├────────────────────────────────────────┤
│                                        │
│ 1. Initialize model with random       │
│    weights                             │
│ 2. Train on data:                      │
│    - Forward pass                      │
│    - Compute loss                      │
│    - Backprop (update weights)         │
│    - Repeat 10 epochs                  │
│                                        │
│ 3. After training:                     │
│    torch.save(model.state_dict(),      │
│               'xray_cnn_model.pth')    │
│                                        │
│    state_dict() = dictionary of all    │
│    learned parameter values            │
│                                        │
│    Saved to disk:                      │
│    ├─ Conv1.weight: (32, 3, 3, 3)      │
│    ├─ Conv1.bias: (32,)                │
│    ├─ Conv2.weight: (64, 32, 3, 3)     │
│    ├─ Conv2.bias: (64,)                │
│    ├─ Conv3.weight: (128, 64, 3, 3)    │
│    ├─ Conv3.bias: (128,)               │
│    ├─ Linear1.weight: (512, 32768)     │
│    ├─ Linear1.bias: (512,)             │
│    ├─ Linear2.weight: (3, 512)         │
│    └─ Linear2.bias: (3,)               │
│                                        │
└────────────────────────────────────────┘
                ▼
          [xray_cnn_model.pth]
        (~50-60 MB binary file)
                ▼
┌────────────────────────────────────────┐
│      INFERENCE (main.py)               │
├────────────────────────────────────────┤
│                                        │
│ 1. Initialize model with random       │
│    weights (same architecture)         │
│ 2. Load saved weights:                 │
│    model.load_state_dict(              │
│      torch.load('xray_cnn_model.pth',  │
│                 map_location=device)   │
│    )                                   │
│                                        │
│    state_dict file → Python dictionary │
│    Dictionary keys → Restore exact     │
│    parameter values from training      │
│                                        │
│ 3. Model now has all the learned       │
│    patterns to diagnose diseases       │
│ 4. Call model.eval() for prediction    │
│    mode (fixed weights, no updates)    │
│                                        │
└────────────────────────────────────────┘
```

---

## 8. COMPLETE REQUEST-RESPONSE CYCLE DIAGRAM

```
┌──────────────────────────────────────────────────────────────────────┐
│                        COMPLETE CYCLE                                │
└──────────────────────────────────────────────────────────────────────┘

TIME  │ FRONTEND                     │ BACKEND              │ FILE SYSTEM
      │                              │                      │
 t0   │ User uploads xray.jpg        │                      │
      │ File input event             │                      │
      │ Preview shown                │                      │
      │                              │                      │
 t1   │ User clicks Predict          │                      │
      │ JavaScript preventDefault    │                      │
      │                              │                      │
 t2   │ Create FormData with file    │                      │
      │ "Predicting..." shown        │                      │
      │                              │                      │
 t3   │ fetch() POST request ────────────→ /predict/ route  │
      │ Browser waits for response  │ ↓                     │
      │                              │ await file.read()    │
      │                              │                      │
 t4   │                              │ Image bytes ─────────────→ Memory
      │                              │ Create PIL Image     │
      │                              │                      │
 t5   │                              │ Apply transforms:    │
      │                              │ • Resize (128×128)   │
      │                              │ • Grayscale          │
      │                              │ • ToTensor           │
      │                              │ • Add batch dim      │
      │                              │                      │
 t6   │                              │ Load model (if not   │
      │                              │ already loaded)      │ xray_cnn_model.pth
      │                              │                      │ ────────↓
      │                              │                      │ Weights loaded
      │                              │                      │ to device (GPU/CPU)
      │                              │                      │
 t7   │                              │ model(input_tensor)  │
      │                              │ 3 Conv layers        │
      │                              │ 3 ReLU activations   │
      │                              │ 3 MaxPool layers     │
      │                              │ Flatten              │
      │                              │ 2 Linear layers      │
      │                              │ Output: (1, 3)       │
      │                              │                      │
 t8   │                              │ torch.max() → index  │
      │                              │ Map to class name    │
      │                              │                      │
 t9   │                              │ Return JSON:         │
      │                              │ {"prediction": "..."}│
      │                              │                      │
 t10  │ ←─ JSON response ←───────────│                      │
      │ Parse data                   │                      │
      │ resultText.textContent =     │                      │
      │   "Prediction: Normal"       │                      │
      │ Display to user              │                      │
      │                              │                      │

Total time: ~100-500ms depending on server hardware and network latency
```

---

## 9. KEY CONCEPTS SUMMARY

| Concept | Explanation |
|---------|-------------|
| **Kernel** | Small 3×3 matrix of weights used by Conv layer; learns patterns |
| **Filters/Channels** | Each Conv layer produces multiple feature maps (32, 64, 128) |
| **Activation (ReLU)** | max(0, x); introduces non-linearity so model can learn complex patterns |
| **Pooling** | MaxPool2d(2×2); reduces spatial size by taking max of 2×2 regions |
| **Dropout** | Random neuron disabling during training; prevents overfitting |
| **Flattening** | Convert 2D feature maps to 1D vector for fully connected layers |
| **Logits** | Raw model output before softmax; higher value = higher confidence |
| **Weights/Parameters** | Learnable values in Conv and Linear layers; updated during training |
| **Gradient** | Computed via backprop; tells how much each weight contributed to loss |
| **Optimizer (Adam)** | Algorithm that updates weights in direction that reduces loss |
| **model.eval()** | Disables dropout and sets BatchNorm to eval mode for inference |
| **torch.no_grad()** | Disables gradient tracking to save memory during inference |

---

## 10. QUICK REFERENCE: FILE ROLES

```
Frontend/website.html
├─ Serves UI to user
├─ Handles file upload and preview
├─ Sends POST request with image file
├─ Displays prediction result

Backend/main.py
├─ FastAPI server (http://localhost:8000)
├─ Exposes /predict/ endpoint
├─ Receives image, preprocesses, runs model
├─ Returns JSON prediction
├─ Loads model at startup

Backend/model.py
├─ Defines DiseaseDiagnosisModel class
├─ Architecture: Conv → ReLU → MaxPool → Conv → ... → Linear
├─ Training code (10 epochs on dataset)
├─ Saves trained weights to xray_cnn_model.pth

xray_cnn_model.pth
├─ Binary file containing learned weights
├─ Loaded at inference time
├─ ~16.8M parameters (~60 MB file)
├─ Enables model to make disease predictions
```

---

This diagram covers the complete flow from user interaction through model prediction!
