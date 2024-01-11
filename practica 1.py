class Perceptron:
    def __init__(self, num_inputs, weights=None, threshold=None):
        # Inicializa los pesos y el umbral
        if weights is None:
            self.weights = [0.0] * num_inputs
        else:
            if len(weights) == num_inputs:
                self.weights = weights
            else:
                raise ValueError("El número de pesos debe ser igual al número de entradas")
        
        if threshold is None:
            self.threshold = 0.0
        else:
            self.threshold = threshold

    def activate(self, inputs):
        # Realiza la suma ponderada de las entradas y aplica la función de activación
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs))
        output = 1 if weighted_sum >= self.threshold else 0
        return output

# Ejemplo de uso
if __name__ == "__main__":
    # Crear una instancia de un perceptrón con 2 entradas, pesos y umbral personalizados
    perceptron = Perceptron(2, weights=[0.5, 0.5], threshold=0.5)

    # Comprobar la salida para diferentes combinaciones de entradas
    inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for input_pair in inputs:
        output = perceptron.activate(input_pair)
        print(f"Entradas {input_pair}: Salida {output}")
