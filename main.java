
import java.io.FileNotFoundException;

public class main {

    public static double treino[][] = {
        {5.1, 4.9, 4.7, 4.6, 5.0, 4.8, 4.3, 5.8, 5.7, 5.4, 5.1, 5.7, 5.1, 5.2, 5.5, 4.9, 5.0, 5.5, 4.9, 4.4, 5.1, 5.0, 4.5, 4.4, 5.0, 5.1, 4.8, 5.1, 4.6, 5.0, 6.4, 6.9, 5.9, 6.0, 6.1, 5.6, 6.7, 5.6, 5.8, 6.2, 5.6, 5.9, 6.1, 6.3, 6.1, 6.4, 6.6, 6.8, 6.7, 6.0, 5.7, 5.5, 5.5, 5.0, 5.6, 5.7, 5.7, 6.2, 5.1, 5.7},
        {3.5, 3.0, 3.2, 3.1, 3.6, 3.0, 3.0, 4.0, 4.4, 3.9, 3.5, 3.8, 3.8, 4.1, 4.2, 3.1, 3.2, 3.5, 3.1, 3.0, 3.4, 3.5, 2.3, 3.2, 3.5, 3.8, 3.0, 3.8, 3.2, 3.3, 3.2, 3.1, 3.0, 2.2, 2.9, 2.9, 3.1, 3.0, 2.7, 2.2, 2.5, 3.2, 2.8, 2.5, 2.8, 2.9, 3.0, 2.8, 3.0, 2.9, 2.6, 2.4, 2.4, 2.3, 2.7, 3.0, 2.9, 2.9, 2.5, 2.8},
        {1.4, 1.4, 1.3, 1.5, 1.4, 1.4, 1.1, 1.2, 1.5, 1.3, 1.4, 1.7, 1.5, 1.5, 1.4, 1.5, 1.2, 1.3, 1.5, 1.3, 1.5, 1.3, 1.3, 1.3, 1.6, 1.9, 1.4, 1.6, 1.4, 1.4, 4.5, 4.9, 4.2, 4.0, 4.7, 3.6, 4.4, 4.5, 4.1, 4.5, 3.9, 4.8, 4.0, 4.9, 4.7, 4.3, 4.4, 4.8, 5.0, 4.5, 3.5, 3.8, 3.7, 3.3, 4.2, 4.2, 4.2, 4.3, 3.0, 4.1},
        {0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.2, 0.4, 0.4, 0.3, 0.3, 0.3, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2, 0.2, 0.3, 0.3, 0.2, 0.6, 0.4, 0.3, 0.2, 0.2, 0.2, 1.5, 1.5, 1.5, 1.0, 1.4, 1.3, 1.4, 1.5, 1.0, 1.5, 1.1, 1.8, 1.3, 1.5, 1.2, 1.3, 1.4, 1.4, 1.7, 1.5, 1.0, 1.1, 1.0, 1.0, 1.3, 1.2, 1.3, 1.3, 1.1, 1.3},
        {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1} //bias
    };

    public static double resultados_esperados[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    public static void main(String[] args) throws FileNotFoundException {

        int numNeuronios_CamadaOculta = 6; //valor "n" qualquer
        int numNeuronios_Entrada = 4; //valor fixo

        MLP rede_neural = new MLP(numNeuronios_CamadaOculta, numNeuronios_Entrada);
        rede_neural.treinar(treino, resultados_esperados);

        System.out.println("Numero de epocas: " + rede_neural.num_Epocas());

        System.out.println("Organismo 1:");
        rede_neural.classificar(new double[]{5.4, 3.9, 1.7, 0.4});
        System.out.println("Organismo 2:");
        rede_neural.classificar(new double[]{4.6, 3.4, 1.4, 0.3});
        System.out.println("Organismo 3:");
        rede_neural.classificar(new double[]{5.0, 3.4, 1.5, 0.2});
        System.out.println("Organismo 4:");
        rede_neural.classificar(new double[]{4.4, 2.9, 1.4, 0.2});
        System.out.println("Organismo 5:");
        rede_neural.classificar(new double[]{4.9, 3.1, 1.5, 0.1});
        System.out.println("Organismo 6:");
        rede_neural.classificar(new double[]{5.4, 3.7, 1.5, 0.2});
        System.out.println("Organismo 7:");
        rede_neural.classificar(new double[]{4.8, 3.4, 1.6, 0.2});
        System.out.println("Organismo 8:");
        rede_neural.classificar(new double[]{5.4, 3.4, 1.7, 0.2});
        System.out.println("Organismo 9:");
        rede_neural.classificar(new double[]{5.1, 3.7, 1.5, 0.4});
        System.out.println("Organismo 10:");
        rede_neural.classificar(new double[]{4.6, 3.6, 1.0, 0.2});
        System.out.println("Organismo 11:");
        rede_neural.classificar(new double[]{5.1, 3.3, 1.7, 0.5});
        System.out.println("Organismo 12:");
        rede_neural.classificar(new double[]{4.8, 3.4, 1.9, 0.2});
        System.out.println("Organismo 13:");
        rede_neural.classificar(new double[]{5.0, 3.0, 1.6, 0.2});
        System.out.println("Organismo 14:");
        rede_neural.classificar(new double[]{5.0, 3.4, 1.6, 0.4});
        System.out.println("Organismo 15:");
        rede_neural.classificar(new double[]{5.2, 3.5, 1.5, 0.2});
        System.out.println("Organismo 16:");
        rede_neural.classificar(new double[]{5.2, 3.4, 1.4, 0.2});
        System.out.println("Organismo 17:");
        rede_neural.classificar(new double[]{4.7, 3.2, 1.6, 0.2});
        System.out.println("Organismo 18:");
        rede_neural.classificar(new double[]{4.8, 3.1, 1.6, 0.2});
        System.out.println("Organismo 19:");
        rede_neural.classificar(new double[]{5.4, 3.4, 1.5, 0.4});
        System.out.println("Organismo 20:");
        rede_neural.classificar(new double[]{5.3, 3.7, 1.5, 0.2});
        System.out.println("Organismo 21:");
        rede_neural.classificar(new double[]{5.5, 2.3, 4.0, 1.3});
        System.out.println("Organismo 22:");
        rede_neural.classificar(new double[]{6.5, 2.8, 4.6, 1.5});
        System.out.println("Organismo 23:");
        rede_neural.classificar(new double[]{5.7, 2.8, 4.5, 1.3});
        System.out.println("Organismo 24:");
        rede_neural.classificar(new double[]{6.3, 3.3, 4.7, 1.6});
        System.out.println("Organismo 25:");
        rede_neural.classificar(new double[]{4.9, 2.4, 3.3, 1.0});
        System.out.println("Organismo 26:");
        rede_neural.classificar(new double[]{6.6, 2.9, 4.6, 1.3});
        System.out.println("Organismo 27:");
        rede_neural.classificar(new double[]{5.2, 2.7, 3.9, 1.4});
        System.out.println("Organismo 28:");
        rede_neural.classificar(new double[]{5.0, 2.0, 3.5, 1.0});
        System.out.println("Organismo 29:");
        rede_neural.classificar(new double[]{6.0, 2.7, 5.1, 1.6});
        System.out.println("Organismo 30:");
        rede_neural.classificar(new double[]{5.4, 3.0, 4.5, 1.5});
        System.out.println("Organismo 31:");
        rede_neural.classificar(new double[]{6.0, 3.4, 4.5, 1.6});
        System.out.println("Organismo 32:");
        rede_neural.classificar(new double[]{6.7, 3.1, 4.7, 1.5});
        System.out.println("Organismo 33:");
        rede_neural.classificar(new double[]{5.8, 2.7, 3.9, 1.2});
        System.out.println("Organismo 34:");
        rede_neural.classificar(new double[]{6.3, 2.3, 4.4, 1.3});
        System.out.println("Organismo 35:");
        rede_neural.classificar(new double[]{5.6, 3.0, 4.1, 1.3});
        System.out.println("Organismo 36:");
        rede_neural.classificar(new double[]{5.5, 2.5, 4.0, 1.3});
        System.out.println("Organismo 37:");
        rede_neural.classificar(new double[]{5.5, 2.6, 4.4, 1.2});
        System.out.println("Organismo 38:");
        rede_neural.classificar(new double[]{6.1, 3.0, 4.6, 1.4});
        System.out.println("Organismo 39:");
        rede_neural.classificar(new double[]{5.8, 2.6, 4.0, 1.2});
        System.out.println("Organismo 40:");
        rede_neural.classificar(new double[]{7.0, 3.2, 4.7, 1.4});
    }

}
