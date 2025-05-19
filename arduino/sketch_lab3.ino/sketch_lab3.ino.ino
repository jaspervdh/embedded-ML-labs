#include <TensorFlowLite.h>

// Include memory measurement utilities
#include "memory_free.h"

// Include the model data and test samples
#include "mnist_model.h"  // Binary model data
#include "testsamples.h"                // Test samples

// TensorFlow Lite Micro headers
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Globals for TensorFlow Lite
namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  
  // Arena size - adjust if needed based on memory errors
  constexpr int kTensorArenaSize = 70 * 1024;  // 70KB - adjust as needed
  alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}

// func for running inference on a sample
int inference(const signed char* test_sample) {

  unsigned long prediction_start_time = micros();
  // copy sample to input tensor
  for (int i = 0; i < 784; i++) {
    input->data.int8[i] = test_sample[i];
  }
  
  // for inference time measurement
  unsigned long start_time = micros();
  
  // actual inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  
  // end and calc inference time measurement
  unsigned long end_time = micros();

  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed");
    return -1;
  }
  
  // extracting output result
  int8_t* output_data = output->data.int8;
  
  // search index with highest probability to print prediction
  int predicted_digit = 0;
  int8_t max_score = output_data[0];
  
  for (int i = 1; i < 10; i++) {
    if (output_data[i] > max_score) {
      predicted_digit = i;
      max_score = output_data[i];
    }
  }

  unsigned long prediction_end_time = micros();

  unsigned long inference_time = end_time - start_time;
  unsigned long prediction_time = prediction_end_time - prediction_start_time;

  // logging inference duration
  Serial.print("Inference time: ");
  Serial.print(inference_time);
  Serial.println(" microseconds");

  Serial.print("Prediction time: ");
  Serial.print(prediction_time);
  Serial.println(" microseconds");
  
  return predicted_digit;
}

void setup() {
  // Initialize serial
  Serial.begin(9600);
  while (!Serial) {
    ; // Wait for serial port to connect
  }
  
  // memory info before model initialization
  Serial.print("Free memory before model initialization: ");
  Serial.print(freeMemory());
  Serial.println(" bytes");
  
  // Initialize TensorFlow Lite
  tflite::InitializeTarget();
  
  // Map the model into a usable data structure
  model = tflite::GetModel(Models_mnist_pruned85_quantint8_tflite);
  
  // Check model version
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Model provided is schema version ");
    Serial.print(model->version());
    Serial.print(" not equal to supported version ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    return;
  }
  
  // This pulls in all the operation implementations we need
  static tflite::AllOpsResolver resolver;
  
  // Build an interpreter to run the model
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  
  // Allocate memory from the tensor_arena for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return;
  }
  
  // get pointers to the model input/output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
  
  // log model input/output details
  Serial.print("Input tensor dimensions: ");
  Serial.print(input->dims->size);
  Serial.print(" dimensions with shape: [");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    if (i < input->dims->size - 1) Serial.print(", ");
  }
  Serial.println("]");
  
  // memory info after model initialization
  Serial.print("Free memory after model initialization: ");
  Serial.print(freeMemory());
  Serial.println(" bytes");
  
  // calculate memory consumption
  Serial.print("Memory consumed by model: ");
  Serial.print(Models_mnist_pruned85_quantint8_tflite_len);
  Serial.println(" bytes");
  

  // inference on each test sample
  Serial.println("\nInferences on test samples:");

  Serial.println("\nTest Sample 0:");
  int prediction = inference(test_sample_0);
  Serial.print("Predicted digit: ");
  Serial.println(prediction);
  
  Serial.println("\nTest Sample 1:");
  prediction = inference(test_sample_1);
  Serial.print("Predicted digit: ");
  Serial.println(prediction);
  
  Serial.println("\nTest Sample 2:");
  prediction = inference(test_sample_2);
  Serial.print("Predicted digit: ");
  Serial.println(prediction);
  
  Serial.println("\nTest Sample 3:");
  prediction = inference(test_sample_3);
  Serial.print("Predicted digit: ");
  Serial.println(prediction);
  
  Serial.println("\nTest Sample 4:");
  prediction = inference(test_sample_4);
  Serial.print("Predicted digit: ");
  Serial.println(prediction);
}

void loop() {
  delay(10000000); // avoid spam
}

// Function to get free memory is defined in memory_free.h