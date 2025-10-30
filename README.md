# ⚡ Spiking Neural Network for IBM DVS Gesture Recognition

This project implements a **Spiking Neural Network (SNN)** to recognize dynamic hand and arm gestures from the **IBM DVS Gesture Dataset**.  
The DVS (Dynamic Vision Sensor) captures motion as asynchronous **spike events** instead of traditional image frames, making SNNs — which process spikes over time — an ideal fit for this task.

---

## 🧠 Overview

- **Objective:** Classify 11 human gestures using event-based DVS data.
- **Dataset:** IBM DVS Gesture Dataset (event streams recorded using a neuromorphic camera).
- **Model Type:** Convolutional Spiking Neural Network (SNN) using **snnTorch**.
- **Framework:** PyTorch.

---

## 🧩 Architecture Summary

| Layer | Type | Description |
|-------|------|-------------|
| Conv2d(2 → 12, 5×5) | Convolution | Extracts low-level motion edges |
| MaxPool2d(2×2) | Pooling | Downsamples feature maps |
| Leaky Integrate-and-Fire | Spiking Layer | Converts activations into spikes |
| Conv2d(12 → 32, 5×5) | Convolution | Learns higher-level motion features |
| MaxPool2d(2×2) | Pooling | Further reduces spatial size |
| Leaky Integrate-and-Fire | Spiking Layer | Adds temporal dynamics |
| Flatten → Linear(800 → 11) | Fully Connected | Classifies into 11 gesture classes |
| Output LIF | Output Layer | Produces spike counts for prediction |

---

## ⚡ Why SNN Instead of CNN?

- **SNNs** handle **temporal and event-based** data more efficiently.
- Operate in an **event-driven** manner → lower power usage.
- Naturally suited for **neuromorphic sensors** like DVS cameras.

---

## 📊 Results

📈 **Graphs and Results**
  
![Accuracy Graph](images/1.png)

---

### 🔹 Confusion Matrix

Visualizes model performance across all gesture classes.

![Confusion Matrix](images/2.png)

---


