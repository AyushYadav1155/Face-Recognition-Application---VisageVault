package com.example.visagevault.face_recognition;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Trace;
import android.util.Pair;


//import com.example.facerecognitionimages.MainActivity;

import com.example.visagevault.MainActivity;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TFLiteFaceRecognition
        implements FaceClassifier {

    //private static final int OUTPUT_SIZE = 512;
    private static final int OUTPUT_SIZE = 512;

    // Only return this many results.
    private static final int NUM_DETECTIONS = 1;

    // Float model
    private static final float IMAGE_MEAN = 128.0f;
    private static final float IMAGE_STD = 128.0f;

    private boolean isModelQuantized;
    // Config values.
    private int inputSize;

    private int[] intValues;

    private float[][] embeedings;

    private ByteBuffer imgData;

    private Interpreter tfLite;



    public void register(String name, Recognition rec) {
        MainActivity.registered.put(name, rec);
    }

    private TFLiteFaceRecognition() {}


    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }



    public static FaceClassifier create(
            final AssetManager assetManager,
            final String modelFilename,
            final int inputSize,
            final boolean isQuantized)
            throws IOException {

        final TFLiteFaceRecognition d = new TFLiteFaceRecognition();
        d.inputSize = inputSize;

        try {
            d.tfLite = new Interpreter(loadModelFile(assetManager, modelFilename));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        d.isModelQuantized = isQuantized;
        // Pre-allocate buffers.
        int numBytesPerChannel;
        if (isQuantized) {
            numBytesPerChannel = 1; // Quantized
        } else {
            numBytesPerChannel = 4; // Floating point
        }
        d.imgData = ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel);
        d.imgData.order(ByteOrder.nativeOrder());
        d.intValues = new int[d.inputSize * d.inputSize];
        return d;
    }


    private Pair<String, Float> findNearest(float[] emb) {
        Pair<String, Float> ret = null;
        for (Map.Entry<String, Recognition> entry : MainActivity.registered.entrySet()) {
            final String name = entry.getKey();
            final float[] knownEmb = ((float[][]) entry.getValue().getEmbeeding())[0];

            float distance = 0;
            for (int i = 0; i < emb.length; i++) {
                float diff = emb[i] - knownEmb[i];
                distance += diff*diff;
            }
            distance = (float) Math.sqrt(distance);
            if (ret == null || distance < ret.second) {
                ret = new Pair<>(name, distance);
            }
        }
        return ret;
    }



    @Override
    public Recognition recognizeImage(final Bitmap bitmap, boolean storeExtra) {
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        imgData.rewind();
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                int pixelValue = intValues[i * inputSize + j];
                if (isModelQuantized) {
                    // Quantized model
                    imgData.put((byte) ((pixelValue >> 16) & 0xFF));
                    imgData.put((byte) ((pixelValue >> 8) & 0xFF));
                    imgData.put((byte) (pixelValue & 0xFF));
                } else { // Float model
                    imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                }
            }
        }
        Object[] inputArray = {imgData};
        // Here outputMap is changed to fit the Face Mask detector
        Map<Integer, Object> outputMap = new HashMap<>();

        embeedings = new float[1][OUTPUT_SIZE];
        outputMap.put(0, embeedings);

        // Run the inference call.
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);


        float distance = Float.MAX_VALUE;
        String id = "0";
        String label = "?";

        if (MainActivity.registered.size() > 0) {
            final Pair<String, Float> nearest = findNearest(embeedings[0]);
            if (nearest != null) {
                final String name = nearest.first;
                label = name;
                distance = nearest.second;
            }
        }
        final int numDetectionsOutput = 1;
        Recognition rec = new Recognition(
                id,
                label,
                distance,
                new RectF());


        if (storeExtra) {
            rec.setEmbeeding(embeedings);
        }

        return rec;
    }


}
