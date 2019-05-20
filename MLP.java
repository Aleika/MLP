
import java.util.Arrays;

public class MLP {

    private double taxa_aprendizado = 0.3;
    private double[][] pesos_PrimeiraCamada;
    private double[] pesos_SegundaCamada;
    private int numNeuronios_PrimeiraCamada;
    private int numNeuronios_Entrada;
    private int epocas = 0;

    public MLP(int numNeuronios_PrimeiraCamada, int numNeuronios_Entrada) {
        this.numNeuronios_PrimeiraCamada = numNeuronios_PrimeiraCamada;
        this.numNeuronios_Entrada = numNeuronios_Entrada;
        calcularPesos();
    }

    public void treinar(double[][] conjuntoTreinamento, double[] valoresEsperados) {
        double erro = 1;
        while ((Math.abs(erro) > 0.01) && (epocas < 10000)) {
            for (int i = 0; i < conjuntoTreinamento[0].length; i++) {
                double[] entradaSegundaCamada = propagation_PrimeiraCamada(conjuntoTreinamento, i);
                double valorSaida = propagation_SegundaCamada(entradaSegundaCamada);
                erro = calcularErro(valoresEsperados[i], valorSaida);
                double g = gradiente(valorSaida, erro);
                aprender(conjuntoTreinamento, entradaSegundaCamada, g, i);
            }
            epocas++;
        }
    }

    public void classificar(double[] entrada) {
        if (epocas > 9999) {
            System.out.println("Nao foi possivel atingir um ponto de convergencia!");
        } else {
            double[] saidasPrimeiraCamada = saidaClassificacao_PrimeiraCamada(entrada);
            double[] entradaSegundaCamada = entradas_SegundaCamada(saidasPrimeiraCamada);
            double y = propagation_SegundaCamada(entradaSegundaCamada);
            int saida = (int) Math.round(y);
            System.out.println(saida);
        }
    }

    private void aprender(double[][] conjuntoTreinamento, double[] entradaSegundaCamada, double gradiente, int i) {
        backpropagation_PrimeiraCamada(conjuntoTreinamento, entradaSegundaCamada, gradiente, i);
        backpropagation_SegundaCamada(entradaSegundaCamada, gradiente);
    }

    private double[] propagation_PrimeiraCamada(double[][] conjuntoTreinamento, int i) {
        double[] saidasPrimeiraCamada = saidaTreinamento_PrimeiraCamada(conjuntoTreinamento, i);
        return entradas_SegundaCamada(saidasPrimeiraCamada);
    }

    private double propagation_SegundaCamada(double[] entradaSegundaCamada) {
        double u = 0;
        for (int i = 0; i< pesos_SegundaCamada.length; i++) {
            u += entradaSegundaCamada[i] * pesos_SegundaCamada[i];
        }
        return sigmoide(u);
    }

    private double[] entradas_SegundaCamada(double[] saidasPrimeiraCamada) {
        double[] entradaSegundaCamada = Arrays.copyOf(saidasPrimeiraCamada, saidasPrimeiraCamada.length + 1);
        entradaSegundaCamada[entradaSegundaCamada.length - 1] = 1.0;
        return entradaSegundaCamada;
    }

    private double[] saidaTreinamento_PrimeiraCamada(double[][] conjuntoTreinamento, int i) {
        double[] saidasPrimeiraCamada = new double[numNeuronios_PrimeiraCamada];
        for (int j = 0; j < pesos_PrimeiraCamada.length; j++) {
            double u = 0;
            for (int k = 0; k < pesos_PrimeiraCamada[j].length; k++) {
                u += conjuntoTreinamento[k][i] * pesos_PrimeiraCamada[j][k];
            }
            saidasPrimeiraCamada[j] = sigmoide(u);
        }
        return saidasPrimeiraCamada;
    }

    private double[] saidaClassificacao_PrimeiraCamada(double[] entrada) {
        double[] saidasPrimeiraCamada = new double[numNeuronios_PrimeiraCamada];
        for (int i = 0; i < pesos_PrimeiraCamada.length; i++) {
            double u = 0;
            for (int j = 0; j< pesos_PrimeiraCamada[i].length; j++) {
                u += entrada[j] * pesos_PrimeiraCamada[i][j];
            }
            saidasPrimeiraCamada[i] = sigmoide(u);
        }
        return saidasPrimeiraCamada;
    }

    private void backpropagation_PrimeiraCamada(double[][] conjuntoTreinamento, double[] entradaSegundaCamada, double gradiente, int i) {
        for (int j = 0; j < entradaSegundaCamada.length - 1; j++) {
            double derivada = derivada_sigmoide(entradaSegundaCamada[j]);
            double sigma = derivada * (pesos_SegundaCamada[j] * gradiente);
            for (int k = 0; k < pesos_PrimeiraCamada[j].length; k++) {
                pesos_PrimeiraCamada[j][k] += taxa_aprendizado * sigma * conjuntoTreinamento[k][i];
            }
        }
    }

    private void backpropagation_SegundaCamada(double[] entradaSegundaCamada, double gradiente) {
        for (int i = 0; i < pesos_SegundaCamada.length; i++) {
            pesos_SegundaCamada[i] += taxa_aprendizado * entradaSegundaCamada[i] * gradiente;
        }
    }

    private double gradiente(double valorSaida, double erro) {
        return valorSaida * (1 - valorSaida) * erro;
    }

    public int num_Epocas() {
        return epocas;
    }

    private double sigmoide(double u) {
        return 1.0 / (1.0 + Math.exp(-u));
    }

    private double derivada_sigmoide(double v) {
        return v * (1.0 - v);
    }

    private double calcularErro(double valorEsperado, double valorSaida) {
        return valorEsperado - valorSaida;
    }

    private void calcularPesos() {
        pesos_PrimeiraCamada = new double[numNeuronios_PrimeiraCamada][numNeuronios_Entrada];
        for (int i = 0; i < pesos_PrimeiraCamada.length; i++) {
            for (int j = 0; j < pesos_PrimeiraCamada[i].length; j++) {
                pesos_PrimeiraCamada[i][j] = Math.random();
            }
        }
        
        pesos_SegundaCamada = new double[numNeuronios_PrimeiraCamada + 1];
        for (int i = 0; i < pesos_SegundaCamada.length; i++) {
            pesos_SegundaCamada[i] = Math.random();
        }
        
    }

    public void imprimirPesos() {
        System.out.println("Pesos da primeira camada:");
        for (int i = 0; i < pesos_PrimeiraCamada.length; i++) {
            for (int j = 0; j < pesos_PrimeiraCamada[i].length; j++) {
                System.out.println(pesos_PrimeiraCamada[i][j]);
            }
        }

        System.out.println("Pesos da segunda camada:");
        for (int i = 0; i < pesos_SegundaCamada.length; i++) {
            System.out.println(pesos_SegundaCamada[i]);
        }

    }


}
