import time

chat_example1 = [
    {"role": "user", "content": "What's the weather like today?"},
    {"role": "assistant", "content": "It's sunny and warm."},
    {"role": "user", "content": "I like the weather. but"},
    {"role": "user", "content": "I'm not sure what to"}
]

chat_example2 = [
    {"role": "user", "content": "आज मौसम कैसा है?"},
    {"role": "assistant", "content": "आज धूप है और गर्मी है।"},
    {"role": "user", "content": "मुझे मौसम अच्छा लगता है, लेकिन"},
    {"role": "user", "content": "मुझे यह नहीं पता कि क्या"}
]

chat_example3 = [
    {"role": "user", "content": "What's the weather like today?"},
    {"role": "assistant", "content": "It's sunny and warm."},
    {"role": "user", "content": "I like the weather. but"},
    {"role": "user", "content": "I'm not sure what to do? But maybe"}
]

UNLIKELY_THRESHOLD = 0.15

def main():
    # Keep optional ONNX/Transformers dependencies out of unittest discovery.
    from backend.src.turn_detection.td import EndOfTurnPredictor

    predictor = EndOfTurnPredictor()

    examples = [chat_example1, chat_example2, chat_example3]
    start_time = time.time()

    for i, example in enumerate(examples, 1):
        prob = predictor.predict(example)
        print(f"End of turn probability {i}: {prob:.4f}")

    print(f"If probability is less than {UNLIKELY_THRESHOLD}, "
          "the model predicts the user hasn't finished speaking.")
    print(f"Prediction time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
