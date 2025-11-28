public class Main {

    public static void main(String[] args) {
        // --- CÓDIGO INICIAL (Declaraciones y Datos XOR) ---
        int i, j; //variables para los bucles for
        double b = 1.000; //valor general de una bia
        int ent = 2; //Entrada de datos

        //Se agregan las épocas y dentro, los datos: bias(col 1), e1(col 2), e2(col 3) y d(col 4). Operación XOR:
        // Bia(s), Entrada(s), Sal. Deseada(s)
        double[][] datos = {{b, 0.0000, 0.0000, 0.0000},
                {b, 0.0000, 1.0000, 1.0000},
                {b, 1.0000, 0.0000, 1.0000},
                {b, 1.0000, 1.0000, 0.0000}
        };

        long repet = 0;
        int epoca = 0;
        double alpha = 0.25; //Tasa de aprendizaje
        int marErrRed = 1; //margen de error para ver si la red aprendió con su salida deseada (d)
        boolean flagPres = true;


        // --- ASIGNACIÓN DE NEURONAS (Capas Oculta y Salida) ---

        //ASIGNACIÓN DE NEURONAS A LA CAPA OCULTA
        //Se asigna 2 neuronas ocultas
        int neuCapOcul = 2;
        //Se crea un arreglo de tipo clsNeurona con neuCapOcul elem.
        clsNeurona[] o = new clsNeurona[neuCapOcul];
        for(i=0;i<o.length;i++){
            //e[i] contiene bias + entradas + Salidas (se resta 1 al no incluir a las salidas)
            o[i] = new clsNeurona(datos[0].length-1);
            //Se Generan sus pesos aleatorios
            o[i].ini_w();
        }

        //ASIGNACIÓN DE NEURONA(S) A LA CAPA DE SALIDA
        //Se asigna 1 neurona de salida
        int neuCapSal = 1;
        //Se crea un arreglo de tipo clsNeurona con neuCapSal elem.
        clsNeurona[] s = new clsNeurona[neuCapSal];
        for(i=0;i<s.length;i++){
            //se suma 1, puesto que se debe agregar la bia y sus entradas de sus neuronas ocultas
            s[i] = new clsNeurona(o.length+1);
            //Se Generan sus pesos aleatorios
            s[i].ini_w();
        }


        // --- BUCLE DE ENTRENAMIENTO PRINCIPAL (while(repet >= 0)) ---
        while(repet>=0){
            //Este bucle indicará cuantas veces se actualizarán los pesos hasta que
            //|fa(s)| este muy cerca de la Sal. Deseada
            System.out.print("\n------------------ Repetición " + repet + " - Comparación sal --> d ------------------\n");

            //Se asume que son 4 épocas [0..3], basándose en sus filas del arreglo datos()
            while(epoca < datos.length){

                //*************
                //PRIMERA FASE (FORWARD)
                //*************

                //Generando la suma ponderada y función de activación en las neuronas ocultas
                for(i=0;i<o.length;i++){
                    o[i].sumPon(datos[epoca]);
                    o[i].funAct();
                }

                //Generando la suma ponderada y función de activación en la neurona de salida
                for(i=0;i<s.length;i++){
                    //enviando bia y neuronas ocultas
                    s[i].sumPon(b, o);
                    s[i].funAct();
                }

                //Se hace la comparación con los datos (columna para "d") vs s[0].fa, bajo un margen de Error (marErrRed)
                System.out.println("Época " + epoca + ": " + precision(s[0].fa, marErrRed) + " --> " +
                        precision(datos[epoca][ent+1], marErrRed));

                //Debe evaluarse que "TODAS" las épocas deben tener el valor TRUE, para finalizar
                flagPres = (flagPres &&
                        (precision(s[0].fa, marErrRed) == precision(datos[epoca][ent+1], marErrRed)));
                //Esta línea debe ir dentro del bucle de épocas, junto con la comparación.


                //*************
                //SEGUNDA FASE (BACKWARD)
                //*************

                //Generando la Delta de la neurona de salida (en este caso solo es 1 neurona)
                for(i=0;i<s.length;i++){
                    s[i].entDeltaSal(datos[epoca][datos[0].length-1]); //enviando desde datos[0][], el valor de la ultima columna
                }

                //Generando las Deltas de las neuronas ocultas
                for(i=0;i<o.length;i++){
                    o[i].entDeltaOcul(s, i+1); //i+1 debido a que las Deltas ocultas no comprometen el uso de las bias
                }

                //Actualizando los pesos que llegan a la neurona salida
                for(i=0;i<s.length;i++){
                    s[i].actWSal(b, alpha, o);
                }

                //Actualizando los pesos que llegan a las neuronas ocultas
                for(i=0;i<o.length;i++){
                    o[i].actWOcul(alpha, datos[epoca]);
                }

                epoca++; // Fin del bucle interno (épocas)
            } // Fin de while(epoca < datos.length)


            // --- VERIFICACIÓN DE ENTRENAMIENTO Y SALIDA ---

            if(flagPres){
                System.out.println("\n------------- FINALIZADO en " + repet + " Repetición(es) -------------");

                //Impresión de pesos finales
                for(i=0;i<o.length;i++){
                    System.out.println("Neurona o" + (i+1));
                    System.out.println("--------------");
                    for(j=0;j<o[i].w.length;j++){
                        if (j == 0){
                            System.out.println("w-bia --> " + o[i].w[j]);
                        }
                        else{
                            System.out.println("w-e" + (j) + "--> " + o[i].w[j]);
                        }
                    }
                }

                for(i=0;i<s.length;i++){
                    System.out.println("Neurona s" + (i+1));
                    System.out.println("--------------");
                    for(j=0;j<s[i].w.length;j++){
                        if (j == 0){
                            System.out.println("w-bia --> " + s[i].w[j]);
                        }
                        else{
                            System.out.println("w-o" + (j) + "--> " + s[i].w[j]);
                        }
                    }
                }

                // Punto de quiebre en caso de éxito
                repet = -1;

            } else {
                //Imprimir pasos de neuronas - Repetición actual
                System.out.println("\n------------- Repetición " + repet + " -------------");

                //Impresión de pesos intermedios de capa oculta
                for(i=0;i<o.length;i++){
                    System.out.println("Neurona o" + (i+1));
                    System.out.println("--------------");
                    for(j=0;j<o[i].w.length;j++){
                        if (j == 0){
                            System.out.println("w-bia --> " + o[i].w[j]);
                        }
                        else{
                            System.out.println("w-e" + (j) + "--> " + o[i].w[j]);
                        }
                    }
                }

                //Impresión de pesos intermedios de capa de salida
                for(i=0;i<s.length;i++){
                    System.out.println("Neurona s" + (i+1));
                    System.out.println("--------------");
                    for(j=0;j<s[i].w.length;j++){
                        if (j == 0){
                            System.out.println("w-bia --> " + s[i].w[j]);
                        }
                        else{
                            System.out.println("w-o" + (j) + "--> " + s[i].w[j]);
                        }
                    }
                }

                //Límite de repeticiones (prevención de bucle infinito)
                if(repet == 100000){ // punto de quiebre en caso no de tener hallazgos de comparación
                    System.out.println("\n------------- SE LLEGÓ AL LÍMITE, SIN HALLAZGO COMPARATIVO -------------");
                    repet = -1;
                }
                else{
                    repet++;
                }

                epoca = 0; // Se resetea la época para la siguiente repetición
                flagPres = true; // Se resetea la bandera para la siguiente repetición
            }
        } // Fin de while(repet >= 0)
    } // Fin de main

    // --- FUNCIÓN AUXILIAR ---

    static double precision(double numero, int digitos) {
        int aux = 1;
        for (int i=0;i<digitos;i++){
            aux = aux * 10;
        }
        return Math.floor(numero * aux) / aux;
    }
}
