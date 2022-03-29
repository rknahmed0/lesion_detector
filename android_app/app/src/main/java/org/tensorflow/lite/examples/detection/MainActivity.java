package org.tensorflow.lite.examples.detection;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.ActivityManager;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ConfigurationInfo;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.os.Bundle;
import android.os.Handler;
import android.os.SystemClock;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.Toast;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.env.Utils;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.YoloV5Classifier;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.Random; // SRA Added

public class MainActivity extends AppCompatActivity {

    // Minimum detection confidence to track a detection.
    public static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.3f;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main);

        cameraButton = findViewById(R.id.cameraButton);
        detectButton = findViewById(R.id.detectButton);
        mProgressBar = (ProgressBar) findViewById(R.id.progressBar);
        imageView = findViewById(R.id.imageView);

        // SRA added QTIM logo
//        qtimLogo = findViewById(R.id.QTIMLogo);
//        this.qtimBitmap = Utils.getBitmapFromAsset(MainActivity.this, "qtim_logo.jpg");
//        this.qtimLogo.setImageBitmap(qtimBitmap);
        // SRA added QTIM logo

        cameraButton.setOnClickListener(v -> startActivity(new Intent(MainActivity.this, DetectorActivity.class)));

        detectButton.setOnClickListener(v -> {
            Handler handler = new Handler();
            detectButton.setEnabled(false); // SRA added
            mProgressBar.setVisibility(ProgressBar.VISIBLE); // SRA progress bar
            detectButton.setText(getString(R.string.run_model)); // SRA running model indicator
            new Thread(() -> {
                final long startTime = SystemClock.uptimeMillis(); // SRA added
//                detector.useGpu(); // SRA added
//                detector.useNNAPI(); // SRA added
                final List<Classifier.Recognition> results = detector.recognizeImage(cropBitmap);
                final long inferenceTime = SystemClock.uptimeMillis() - startTime; // SRA added
                Log.d("YoloV5Classifier",  "inference time (ms): " + inferenceTime); // SRA added
                handler.post(new Runnable() {
                    @Override
                    public void run() {
                        handleResult(cropBitmap, results);
                    }
                });
            }).start();

        });

        // SRA
        this.sourceBitmap = Utils.getBitmapFromAsset(MainActivity.this, mTestImages[mImageIndex]);
        this.cropBitmap = Utils.processBitmap(sourceBitmap, TF_OD_API_INPUT_SIZE);
        this.imageView.setImageBitmap(cropBitmap);

        final Button testButton = findViewById(R.id.testButton);
        testButton.setText(("Sample Image 1/6"));
        testButton.setOnClickListener(v -> {
            trackingOverlay.setVisibility(View.INVISIBLE);
            mImageIndex = (mImageIndex + 1) % mTestImages.length;
            testButton.setText(String.format("Sample Image %d/%d", mImageIndex + 1, mTestImages.length));
            this.sourceBitmap = Utils.getBitmapFromAsset(MainActivity.this, mTestImages[mImageIndex]);
            this.cropBitmap = Utils.processBitmap(sourceBitmap, TF_OD_API_INPUT_SIZE);
            this.imageView.setImageBitmap(cropBitmap);
        });
        // SRA

        initBox();
        ActivityManager activityManager = (ActivityManager) getSystemService(Context.ACTIVITY_SERVICE);
        ConfigurationInfo configurationInfo = activityManager.getDeviceConfigurationInfo();

        System.err.println(Double.parseDouble(configurationInfo.getGlEsVersion()));
        System.err.println(configurationInfo.reqGlEsVersion >= 0x30000);
        System.err.println(String.format("%X", configurationInfo.reqGlEsVersion));
    }

    private static final Logger LOGGER = new Logger();

    public static final int TF_OD_API_INPUT_SIZE = 640;

    private static final boolean TF_OD_API_IS_QUANTIZED = false;

    private static final String TF_OD_API_MODEL_FILE = "yolov5s-fp16.tflite"; // SRA changed from "yolov5s.tflite";

    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/coco.txt"; // coco.txt has only one cervix class

    private static final boolean MAINTAIN_ASPECT = true;
    private Integer sensorOrientation = 90;

//    private Classifier detector; // SRA commented
    private YoloV5Classifier detector; // SRA added

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;
    private MultiBoxTracker tracker;
    private OverlayView trackingOverlay;

    protected int previewWidth = 0;
    protected int previewHeight = 0;

    private Bitmap sourceBitmap;
    private Bitmap cropBitmap;

    private int mImageIndex = 0; // SRA added
    private String[] mTestImages = {"lesion_1.jpg","normal_1.jpg","lesion_2.jpg","normal_2.jpg","lesion_3.jpg","lesion_4.jpg"}; // SRA added

    private Button cameraButton, detectButton;
    private ProgressBar mProgressBar; // SRA added progress bar
    private ImageView imageView;

//    private ImageView qtimLogo;
//    private Bitmap qtimBitmap;

    private void initBox() {
        previewHeight = TF_OD_API_INPUT_SIZE;
        previewWidth = TF_OD_API_INPUT_SIZE;
        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        tracker = new MultiBoxTracker(this);
        trackingOverlay = findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                canvas -> tracker.draw(canvas));

        tracker.setFrameConfiguration(TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, sensorOrientation);

        try {
            detector =
                    YoloV5Classifier.create(
                            getAssets(),
                            TF_OD_API_MODEL_FILE,
                            TF_OD_API_LABELS_FILE,
                            TF_OD_API_IS_QUANTIZED,
                            TF_OD_API_INPUT_SIZE);
            Log.d("YoloV5Classifier",  "model loaded successfully: " + TF_OD_API_MODEL_FILE); // SRA added
//            detector.useGpu(); // SRA added
//            detector.useNNAPI(); // SRA added
            detector.setNumThreads(4); // SRA added
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }
    }

    private void handleResult(Bitmap bitmap, List<Classifier.Recognition> results) {
        final Canvas canvas = new Canvas(bitmap);
        final Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(2.0f);

        final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();

        for (final Classifier.Recognition result : results) {
            final RectF location = result.getLocation();
            if (location != null && result.getConfidence() >= MINIMUM_CONFIDENCE_TF_OD_API) {
//                canvas.drawRect(location, paint);
                cropToFrameTransform.mapRect(location);

                result.setLocation(location);
                mappedRecognitions.add(result);
            }
        }
        tracker.trackResults(mappedRecognitions, new Random().nextInt());
        trackingOverlay.postInvalidate();
        trackingOverlay.setVisibility(View.VISIBLE); // SRA added
        imageView.setImageBitmap(bitmap);
        detectButton.setEnabled(true); // SRA added
        detectButton.setText(getString(R.string.detect)); // SRA added
        mProgressBar.setVisibility(ProgressBar.INVISIBLE); // SRA added
    }
}
