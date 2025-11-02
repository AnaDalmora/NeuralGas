#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <ctime>

const int DIMENSAO = 3; 

using Vetor = std::vector<double>; 

struct Neuronio {
    Vetor w; 
};

struct ItemRanqueado {
    double distancia;
    int indice_neuronio;

    bool operator<(const ItemRanqueado& outro) const {
        return distancia < outro.distancia;
    }
};

double CalcularDistanciaEuclidiana(const Vetor& v1, const Vetor& v2) {
    double soma = 0.0;
    for (int i = 0; i < DIMENSAO; ++i) {
        soma += std::pow(v1[i] - v2[i], 2);
    }
    return soma;
}

double FuncaoVizinhanca(int ranque_k, double lambda) {
    return std::exp(-(static_cast<double>(ranque_k) / lambda));
}

double CalcularDecaimento(double valor_inicial, double valor_final, int iteracao_atual, int max_iteracoes) {
    return valor_inicial - (valor_inicial - valor_final) * (static_cast<double>(iteracao_atual) / max_iteracoes);
}

void InicializarPesos(std::vector<Neuronio>& rede, int numero_neuronios) {
    std::random_device rd;
    std::mt19937 gerador_mt(rd());

    std::uniform_real_distribution<> distribuicao_real(0.0, 1.0); 

    for (int i = 0; i < numero_neuronios; ++i) {
        rede[i].w.resize(DIMENSAO);
        for (int j = 0; j < DIMENSAO; ++j) {
            rede[i].w[j] = distribuicao_real(gerador_mt);
        }
    }
}

Vetor AmostrarDado(const std::vector<Vetor>& conjunto_dados) {
    static std::default_random_engine gerador_amostra(std::time(0));
    std::uniform_int_distribution<int> distribuicao_indice(0, conjunto_dados.size() - 1);
    return conjunto_dados[distribuicao_indice(gerador_amostra)];
}


std::vector<Neuronio> TreinarNeuralGas(
    const std::vector<Vetor>& conjunto_dados,
    int numero_neuronios,
    int max_iteracoes) 
{
    double alfa_inicial = 0.5;
    double lambda_inicial = numero_neuronios / 2.0;
    double alfa_final = 0.005;
    double lambda_final = 0.01;
    
    std::vector<Neuronio> rede(numero_neuronios);
    InicializarPesos(rede, numero_neuronios);

    std::vector<ItemRanqueado> ranqueamento_distancia(numero_neuronios);

    for (int t = 0; t < max_iteracoes; ++t) {
        
        double alfa_atual = CalcularDecaimento(alfa_inicial, alfa_final, t, max_iteracoes);
        double lambda_atual = CalcularDecaimento(lambda_inicial, lambda_final, t, max_iteracoes);
        
        Vetor x = AmostrarDado(conjunto_dados);
        
        for (int i = 0; i < numero_neuronios; ++i) {
            ranqueamento_distancia[i].distancia = CalcularDistanciaEuclidiana(x, rede[i].w);
            ranqueamento_distancia[i].indice_neuronio = i;
        }
        
        std::sort(ranqueamento_distancia.begin(), ranqueamento_distancia.end());

        //Adaptação
        for (int k = 0; k < numero_neuronios; ++k) {
            int indice_neuronio = ranqueamento_distancia[k].indice_neuronio;
            
            double fator_vizinhanca = FuncaoVizinhanca(k, lambda_atual);
            
            for (int j = 0; j < DIMENSAO; ++j) {
                double delta = alfa_atual * fator_vizinhanca * (x[j] - rede[indice_neuronio].w[j]);
                rede[indice_neuronio].w[j] += delta;
            }
        }

        //implementar visualização do treinamento
    }

    return rede;
}


int main() {
    
    std::vector<Vetor> dados_treinamento = {
        {0.9, 0.1, 0.1}, 
        {0.8, 0.2, 0.2},
        {0.1, 0.9, 0.1}, 
        {0.2, 0.8, 0.2},
        {0.5, 0.5, 0.5}, 
        {0.6, 0.6, 0.6},
        {0.1, 0.1, 0.9}, 
        {0.2, 0.2, 0.8},
        {0.15, 0.85, 0.1},
        {0.85, 0.15, 0.1}
    };

    int K_neuronios = 4; 
    int T_iteracoes = 10000; 

    std::cout << "Iniciando treinamento do Neural Gas (K=" << K_neuronios << ", T=" << T_iteracoes << ")..." << std::endl;

    std::vector<Neuronio> rede_treinada = TreinarNeuralGas(dados_treinamento, K_neuronios, T_iteracoes);

    std::cout << "Vetores de Atributo:" << std::endl;

    for (int i = 0; i < rede_treinada.size(); ++i) {
        std::cout << "W[" << i << "]: [";
        for (int j = 0; j < DIMENSAO; ++j) {
            std::cout << rede_treinada[i].w[j] << (j < DIMENSAO - 1 ? ", " : "");
        }
        std::cout << "]" << std::endl;
    }

    return 0;
}