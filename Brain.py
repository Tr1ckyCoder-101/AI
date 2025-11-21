import math
import json


class Brain:
    # model: {label: [float, float, ...]}
    model = {}

    @staticmethod
    def normalizePoints(points):
        if not points:
            return []
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        minX, maxX = min(xs), max(xs)
        minY, maxY = min(ys), max(ys)
        rangeX = maxX - minX or 1
        rangeY = maxY - minY or 1
        # Return a new list, do not modify in-place
        return [[(p[0] - minX) / rangeX, (p[1] - minY) / rangeY] for p in points]

    @staticmethod
    def genNormalMatrix(normalizedPoints):
        res = 10  # Match MLDrawingPad
        matrix = [[0 for _ in range(res)] for _ in range(res)]
        for point in normalizedPoints:
            x = int(point[0] * res)
            y = int(point[1] * res)
            if x >= res:
                x = res - 1
            if y >= res:
                y = res - 1
            matrix[y][x] += 1  # Increment instead of set
        # Normalize matrix to [0,1] range
        max_val = max(max(row) for row in matrix) or 1
        for y in range(res):
            for x in range(res):
                matrix[y][x] = matrix[y][x] / max_val
        return matrix

    @staticmethod
    def flattenMatrix(matrix):
        flatVector = []
        for row in matrix:
            for cell in row:
                flatVector.append(cell)
        return flatVector

    # visualizeMatrix method removed for cleaner output
    # (If needed, you can uncomment and use the following for debugging:)
    # @staticmethod
    # def visualizeMatrix(matrix):
    #     visualization = ""
    #     for row in matrix:
    #         visualization += " ".join(str(cell) for cell in row) + "\n"
    #     print(visualization)

    @staticmethod
    def manhattanDistance(vector1, vector2):
        total_sum = 0
        for i in range(len(vector1)):
            total_sum += abs(vector1[i] - vector2[i])
        return total_sum

    @staticmethod
    def euclideanDist(vec1, vec2):
        dist = 0
        for i in range(len(vec1)):
            diff = vec1[i] - vec2[i]
            dist += diff * diff
        return math.sqrt(dist)

    @staticmethod
    def hammingDist(vec1, vec2):
        # Count the number of differing bits
        return sum(a != b for a, b in zip(vec1, vec2))

    @staticmethod
    def predictObject(predict_vector):
        bestMatch = None
        lowestDistance = float('inf')
        print("[DEBUG] Input vector length:", len(predict_vector))
        for label, modelVector in Brain.model.items():
            if not isinstance(modelVector, list):
                continue
            if len(modelVector) != len(predict_vector):
                print(f"[DEBUG] Skipping '{label}' due to length mismatch.")
                continue
            distance = Brain.euclideanDist(modelVector, predict_vector)
            print(f"[DEBUG] Euclidean distance to '{label}': {distance}")
            if bestMatch is None or distance < lowestDistance:
                lowestDistance = distance
                bestMatch = label
        if bestMatch is not None:
            print(f"[DEBUG] Closest match: {bestMatch} (Euclidean distance {lowestDistance})")
            return bestMatch
        print("[DEBUG] No match found. Returning 'unknown'.")
        return "unknown"

    @staticmethod
    def trainModel(label, vector, learningRate=0.05):
        # In-place learning rate update (like MLDrawingPad)
        vector = [float(x) for x in vector]
        if label not in Brain.model or not isinstance(Brain.model[label], list) or len(Brain.model[label]) != len(vector):
            Brain.model[label] = vector[:]
        else:
            model_vec = Brain.model[label]
            for i in range(len(model_vec)):
                diff = vector[i] - model_vec[i]
                model_vec[i] += learningRate * diff

    @staticmethod
    def packBrainModel(file_path):
        try:
            print(f"[DEBUG] Attempting to load model from: {file_path}")
            with open(file_path, "r") as file:
                loaded = json.load(file)
                # Convert all values to float lists
                for k, v in loaded.items():
                    if isinstance(v, list):
                        loaded[k] = [float(x) for x in v]
                Brain.model = loaded
            print("[DEBUG] Model loaded. Keys:", list(Brain.model.keys()))
        except Exception as e:
            Brain.model = {}
            print(f"[DEBUG] Model load failed: {e}. Initialized empty model.")

    @staticmethod
    def saveModel(file_path):
        try:
            # Save as single vector for each label
            with open(file_path, "w") as file:
                json.dump(Brain.model, file)
            print(f"[DEBUG] Model saved to: {file_path}")
        except Exception as e:
            print(f"[DEBUG] Model save failed: {e}")