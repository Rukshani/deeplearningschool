
# Returns predictions for array of input heights
def predict(H, data):
    return [height > H for height in data]
heights = [20, 40, 41, 45, 48, 50, 60]
allowed = [False, False, False, False, True, True, True]
# Evaluates and returns the current model accuracy
def evaluate(H):
    predictions = predict(H, heights)
    num_correct = 0
    for prediction, target in zip(predictions, allowed):
        if prediction == target: num_correct += 1
    accuracy = 100.0*num_correct/(len(allowed))
    return accuracy

best_H = 0
best_accuracy = 0.0
for H in range(38, 50):
    accuracy = evaluate(H=H)
    print (H, accuracy)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_H = H

print ("")
print ("Best threshold: ", best_H)
print ("Best accuracy: ", best_accuracy)
