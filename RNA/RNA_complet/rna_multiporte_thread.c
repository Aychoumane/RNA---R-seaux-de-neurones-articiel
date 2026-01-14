//gcc rna_multiporte_thread.c -o rna_multiporte_thread -pthread -lm && ./rna_multiporte_thread 

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <pthread.h>

// --------- Paramètres du réseau ---------
#define INPUTS 2
#define HIDDEN 3
#define OUTPUTS 1
#define EPOCHS 10000

// --------- Mutex pour affichage ---------
pthread_mutex_t print_mutex;

// --------- Fonctions utilitaires ---------
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double d_sigmoid(double y) {
    return y * (1.0 - y);
}

double random_weight() {
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

void line() {
    printf("------------------------------\n");
}

// --------- Structure du réseau ---------
typedef struct {
    char name[20];   // Nom du réseau (ex: "AND", "XOR")
    double weight_input_hidden[INPUTS][HIDDEN];
    double bias_hidden[HIDDEN];
    double weight_hidden_output[HIDDEN];
    double bias_output;
    double learning_rate;
} NeuralNetwork;

// --------- Structure pour threads ---------
typedef struct {
    NeuralNetwork *nn;
    double (*inputs)[2];
    double *targets;
    double *epoch_errors; // Tableau pour stocker les erreurs par epoch
} ThreadData;

// --------- Initialisation du réseau ---------
void init_network(NeuralNetwork *nn, const char *name, double learning_rate) {
    strcpy(nn->name, name);
    for(int i = 0; i < INPUTS; i++)
        for(int j = 0; j < HIDDEN; j++)
            nn->weight_input_hidden[i][j] = random_weight();

    for(int j = 0; j < HIDDEN; j++) {
        nn->bias_hidden[j] = random_weight();
        nn->weight_hidden_output[j] = random_weight();
    }

    nn->bias_output = random_weight();
    nn->learning_rate = learning_rate;
}

// --------- Propagation avant ---------

double forward_propagation(NeuralNetwork *nn, double input1, double input2, double hidden[HIDDEN]) {
    // Calcul des neurones cachés
    for (int j = 0; j < HIDDEN; j++) {
        double sum_hidden = input1 * nn->weight_input_hidden[0][j]
                          + input2 * nn->weight_input_hidden[1][j]
                          + nn->bias_hidden[j];
        hidden[j] = sigmoid(sum_hidden);
    }

    // Calcul de la sortie
    double sum_output = 0.0;
    for (int j = 0; j < HIDDEN; j++) {
        sum_output += hidden[j] * nn->weight_hidden_output[j];
    }
    sum_output += nn->bias_output;

    return sigmoid(sum_output);
}
// --------- Rétropropagation ---------
void backpropagation(NeuralNetwork *nn, double input1, double input2, 
                     double hidden[HIDDEN], double output, double target) {

    double error = target - output;

    // Erreur sortie
    double delta_output = error * d_sigmoid(output);

    // Erreurs cachées
    double delta_hidden[HIDDEN];
    for (int j = 0; j < HIDDEN; j++) {
        delta_hidden[j] = d_sigmoid(hidden[j]) 
                        * nn->weight_hidden_output[j] 
                        * delta_output;
    }

    // Mise à jour poids cachée -> sortie
    for (int j = 0; j < HIDDEN; j++) {
        nn->weight_hidden_output[j] += nn->learning_rate * delta_output * hidden[j];
    }
    nn->bias_output += nn->learning_rate * delta_output;

    // Mise à jour poids entrée -> cachée
    for (int j = 0; j < HIDDEN; j++) {
        nn->weight_input_hidden[0][j] += nn->learning_rate * delta_hidden[j] * input1;
        nn->weight_input_hidden[1][j] += nn->learning_rate * delta_hidden[j] * input2;
        nn->bias_hidden[j] += nn->learning_rate * delta_hidden[j];
    }
}
// --------- Entraînement ---------
void *train(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    NeuralNetwork *nn = data->nn;
    double (*inputs)[2] = data->inputs;
    double *targets = data->targets;
    double *errors = data->epoch_errors;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double total_error = 0.0;

        for (int sample = 0; sample < 4; sample++) {
            double input1 = inputs[sample][0];
            double input2 = inputs[sample][1];
            double target = targets[sample];

            double hidden[HIDDEN];
            double output = forward_propagation(nn, input1, input2, hidden);

            double error = target - output;
            total_error += error * error;

            backpropagation(nn, input1, input2, hidden, output, target);
        }

        errors[epoch] = total_error;
    }

    pthread_exit(NULL);
}


// --------- Test et affichage ---------
void test_and_print(NeuralNetwork *nn, double inputs[4][2], double *epoch_errors) {
    int sample, epoch;
    double input1, input2;
    double hidden[HIDDEN];
    double output;

    // Calcul des sorties
    double outputs[4];
    for (sample = 0; sample < 4; sample++) {
        input1 = inputs[sample][0];
        input2 = inputs[sample][1];
        outputs[sample] = forward_propagation(nn, input1, input2, hidden);
    }

    // Affichage protégé par mutex
    pthread_mutex_lock(&print_mutex);
    printf("\nRésultats pour la porte logique %s :\n", nn->name);
    for (sample = 0; sample < 4; sample++) {
        printf("Entrées: %.0f, %.0f => Sortie prédite: %.3f\n",
               inputs[sample][0], inputs[sample][1], outputs[sample]);
    }
    line();

    for (epoch = 0; epoch < EPOCHS; epoch += 2000) {
        printf("[%s] Epoch %d - Erreur totale: %f\n", nn->name, epoch, epoch_errors[epoch]);
    }
    line();
    pthread_mutex_unlock(&print_mutex);
}


// --------- Main ---------
int main() {
    srand(time(NULL));
    pthread_mutex_init(&print_mutex, NULL);

    int nb_gate = 3;

    double inputs[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    double targets[][4] = {
        {0,0,0,1}, // AND
        {0,1,1,1}, // OR
        {0,1,1,0}  // XOR
    };
    const char *names[] = {"AND", "OR", "XOR"};

    NeuralNetwork nns[nb_gate];
    double errors[nb_gate][EPOCHS];
    ThreadData data[nb_gate];
    pthread_t threads[nb_gate];

    for(int i = 0; i < nb_gate; i++) {
        init_network(&nns[i], names[i], 0.1);
        data[i].nn = &nns[i];
        data[i].inputs = inputs;
        data[i].targets = targets[i];
        data[i].epoch_errors = errors[i];
    }

    for(int i = 0; i < nb_gate; i++)
        pthread_create(&threads[i], NULL, train, &data[i]);

    for(int i = 0; i < nb_gate; i++)
        pthread_join(threads[i], NULL);

    for(int i = 0; i < nb_gate; i++)
        test_and_print(&nns[i], inputs, errors[i]);

    pthread_mutex_destroy(&print_mutex);
    return 0;
}
