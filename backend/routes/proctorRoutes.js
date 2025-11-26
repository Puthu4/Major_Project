// backend/routes/proctorRoutes.js

import express from "express";
import * as faceapi from "face-api.js";
import canvas from "canvas";
import path from "path";
import User from "../models/User.js";
import MalpracticeLog from "../models/MalpracticeLog.js";

const router = express.Router();

const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const MODELS_DIR = path.join(process.cwd(), "face-models-tiny");
let modelsLoaded = false;

// Load models once on server start
(async function loadModels() {
  try {
    await faceapi.nets.tinyFaceDetector.loadFromDisk(MODELS_DIR);
    await faceapi.nets.faceLandmark68TinyNet.loadFromDisk(MODELS_DIR);
    await faceapi.nets.faceRecognitionNet.loadFromDisk(MODELS_DIR);
    modelsLoaded = true;
    console.log("FaceAPI models loaded");
  } catch (err) {
    console.error("Failed to load models:", err);
    modelsLoaded = false;
  }
})();

router.post("/check", async (req, res) => {
  try {
    if (!modelsLoaded) {
      return res.json({ status: "error", message: "Models not loaded" });
    }

    const { userId, challengeId, image } = req.body;

    if (!image) {
      return res.json({ status: "error", message: "No image received" });
    }

    // Convert dataURL to Buffer
    const base64Data = image.replace(/^data:image\/\w+;base64,/, "");
    const imgBuffer = Buffer.from(base64Data, "base64");

    const img = await canvas.loadImage(imgBuffer);

    const detection = await faceapi
      .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks(true)
      .withFaceDescriptor();

    if (!detection) {
      return res.json({ status: "noface" });
    }

    // Load user stored descriptor
    const user = await User.findById(userId);
    if (!user || !user.faceDescriptor) {
      return res.json({ status: "error", message: "User has no saved descriptor" });
    }

    const queryDescriptor = detection.descriptor;
    const storedDescriptor = new Float32Array(user.faceDescriptor);
    const distance = faceapi.euclideanDistance(queryDescriptor, storedDescriptor);

    const verified = distance < 0.60; // stricter, realistic

    if (!verified) {
      await MalpracticeLog.create({
        userId,
        challengeId,
        flags: ["Face mismatch"],
      });

      return res.json({ status: "mismatch", distance });
    }
  return res.json({ status: "ok", distance });
  } catch (err) {
    console.error("Proctor check error:", err);
    return res.json({ status: "error", message: "Server error" });
  }
});

export default router;
