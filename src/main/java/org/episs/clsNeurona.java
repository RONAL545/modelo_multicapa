import java.util.Random;
public class clsNeurona {
    double[] w;
    double sp;
    double fa;
    double delta;

    // 2. CONSTRUCTOR FALTANTE (ACEPTA UN INT)
    public clsNeurona(int numEntradas) {
        // Inicializa el array de pesos con el tamaño correcto (Entradas + 1 para el Bias)
        this.w = new double[numEntradas];
    }

    public void ini_w(){
        Random ran = new Random();
        for(int i=0;i<this.w.length;i++){
            this.w[i] = ran.nextDouble() * 2 - 1; //aleatorio entre -1.0 a 1.0
        }
    }

    public void sumPon(double[] e){
        this.sp = 0.0; //inicializo la suma ponderada para calcular con los pesos
        for(int i=0;i<e.length-1;i++){
            this.sp = this.sp + e[i] * this.w[i];
        }
    }

    public void sumPon(double b, clsNeurona[] o){ //Polimorfismo para Sum. Pon. de Capa Oculta
        //Aqui solo se traen las neuronas ocultas por lo que falta su bia, como dato complementario
        this.sp = b * this.w[0];
        for(int i=0;i<o.length;i++){
            this.sp = this.sp + o[i].fa * this.w[i+1];
        }
    }

    public void funAct(){
        this.fa = 1 / (1 + Math.exp(-this.sp));
    }

    public void entDeltaSal(double d){
        //Hallazgo de la Delta de la Neurona si de la Capa de Salida
        this.delta = (d - this.fa) * this.fa * (1 - this.fa);
    }

    // --- FUNCIONES DE LA TERCERA IMAGEN (Actualización de Pesos) ---

    public void entDeltaOcul(clsNeurona[] s, int pos){
        //pos: Posición del peso que llega a la salida.
        //Variable que contendrá la suma ponderada comprometida con la Delta de salida (Regla de la cadena)
        double spDelSal = 0;
        for(int i=0;i<s.length;i++){
            //Se busca el w[pos] de la neurona s[i] multiplicando su delta de s[i]
            spDelSal = spDelSal + (s[i].w[pos] * s[i].delta);
        }

        this.delta = this.fa * (1 - this.fa) * spDelSal; //formula completa de la Delta
    }

    public void actWSal(double b, double alpha, clsNeurona[] o){
        //actualizando peso de la bia
        this.w[0] = this.w[0] + (alpha * this.delta * b);
        //Act. pesos que llegan a la capa oculta (todas sus o[]).
        //i = 1, ya que w[0] fue evaluado con su bia
        for(int i=1;i<this.w.length;i++){
            //Se asume que estamos actualmente en el para evaluar su Delta
            this.w[i] = this.w[i] + (alpha * this.delta * o[i-1].fa);
        }
    }

    public void actWOcul(double alpha, double[] e){
        for(int i=0;i<this.w.length;i++){ //verificando todas las (e) incluida la bia
            //Se asume que estamos actualmente en la neurona o[i] para evaluar su Delta
            this.w[i] = this.w[i] + (alpha * this.delta * e[i]);
        }
    }
}