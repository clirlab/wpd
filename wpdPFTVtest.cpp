//*************************************************************************
//
//    Programa que resuelve el modelo para filtrado utilizando punto fijo
//    realiza K iteraciones de Gauss-Seidel
//
// Author       : Iván de Jesús May-Cen
// Language     : C++
// Compiler     : g++
// Environment  : 
// Revisions
//   Initial    : 2021-02-15 17:36:44 
//   Last       : 2022-07-31
//
//  para compilar
//    g++ -O2 wpdPFTVtest.cpp -o test -lrt -lblitz `pkg-config --cflags opencv4` `pkg-config --libs opencv4`
//  para ejecutar
//    ./test
// 
//*************************************************************************

// preprocesor directives
#include <opencv2/core/core.hpp>                // OpenCV      
#include <opencv2/highgui/highgui.hpp>           
#include <blitz/array.h>                        // Blitz++             
#include <random/uniform.h>
#include <random/normal.h>
#include <sys/time.h>                   // funciones de tiempo
#include <cmath>                        // funciones matematicas
#include <float.h>                      // mathematical constants
#include <iostream>                                  
#include <fstream> 

// declara namespace
using namespace std;
using namespace cv;
using namespace blitz;
using namespace ranlib;

// resolucion de la imagen
//int RENGLONES = 960, COLUMNAS = 1280;
int RENGLONES = 480, COLUMNAS = 640;
//int RENGLONES = 480/2, COLUMNAS = 640/2;
int renglones = RENGLONES, columnas = COLUMNAS;
int TIPO = 9;
int opcion; // variable for switch (0 for sinthetic data, 1 for real data)
Mat IMAGEN;
char name_image[20];
 
// parametros para funcional a utilizar en la optimizacion
const double LAMBDA1 = 1.0;
const double LAMBDA2 = 1.0;
const double LAMBDA3 = 1.0;
const int K = 5;
const double BETA = 1.0e-3;

// parametros para ruido aditivo gaussiano
const double CTE = 0.5, factor = 3.0;

// variables del metodo numerico
const double EPSILON1 = 1.0e-6;         // criterio de paro del algoritmo, gradiente
const double EPSILON2 = 1.0e-6;         // criterio de paro del algoritmo, cambio en x
const unsigned ITER_MAX1 = 200000;    // maximo de iteraciones 

// declaracion de funciones
double funcionPhase(double,double);
void Derivada( Array<double,2>& dIs, Array<double,2>& dIc, Array<double,2>& WP, Array<double,2>& Is, Array<double,2>& Ic );
void iteracion_Gauss_Seidel( Array<double,2>& WP, Array<double,2>& Is, Array<double,2>& Ic, Array<double,2>& Is1, Array<double,2>& Ic1 );
double Funcional( Array<double,2>& WP, Array<double,2>& Is, Array<double,2>& Ic );
double gradientWrap(double,double);
void gradienteArray(Array<double,2>&, Array<double,2>&, Array<double,2>&);
double minMod(double,double);  
void boundaryCond1(Array<double,2>&);
void boundaryCond2(Array<double,2>&);
void Print3D(Array<double,2>,FILE*,const char*);


//*************************************************************************
//
//                        inicia funcion principal
//
//*************************************************************************
int main( int argc, char **argv  )
{
  // read option to data sinthetic(opcion = 0) or data real (opcion = 1)
  opcion = atoi(argv[1]);
  
  while(!(opcion == 0 || opcion == 1)) 
     {
      cout << endl << "Please, enter a valid option:" << endl;
      cout << endl << "0 : Sinthetic image" << endl;
      cout << endl << "1 : Real image" << endl;
      cout << endl << "Enter a valid option : " << endl;
      cin >> opcion;
     }
  //parametros desde consola
  if(opcion == 0) 
    {
     renglones = RENGLONES; // image rows
     columnas = COLUMNAS;  // image cols 
     TIPO = 9;      //option for sinthetic phase
     // despliega informacion del proceso
     cout << endl << "Proccesing data sinthetic..." << endl << endl;
    }
  else if(opcion == 1)
    {
      // name of image archive
      //name_image = "brain.jpeg";
      sprintf(name_image, "brain.jpeg");
      //strcpy(name_image, argv[1]);
      IMAGEN = imread(name_image, IMREAD_GRAYSCALE);//IMREAD_GRAYSCALE
      renglones = IMAGEN.rows, columnas = IMAGEN.cols;
      // despliega informacion del proceso
      cout << endl << "Proccesing data real..." << endl << endl;
    }


  // separa memoria para procesar con Blitz
  NormalUnit<double> ruidoInicial;// Continuous normal distribution with mean = 0.0, variance = 1
//  ruidoInicial.seed( (unsigned int)time(0) );
    ruidoInicial.seed( 0 ); //fija semilla
  Normal<double> ruidoImagen(0.0,CTE);// Continuous normal distribution with mean = 0.0, variance = CTE
//  ruidoImagen.seed( (unsigned int)time(0) );
    ruidoImagen.seed( 0 );// fija semilla  
  Array<double,2> U_im(renglones,columnas), U_re(renglones,columnas), phase0(renglones,columnas), U_im0(renglones,columnas), U_re0(renglones,columnas),
                           dummy(renglones,columnas);
  Array<double,2> P(renglones,columnas), WP(renglones,columnas), WP0(renglones,columnas), Po(renglones,columnas);

  Array<double,2> U_imo(renglones,columnas), U_reo(renglones,columnas);

  Array<double,2> U_im1(renglones,columnas), dIs(renglones,columnas), U_re1(renglones,columnas), dIc(renglones,columnas), Paux(renglones,columnas);
  // crea manejador de imagenes con openCV
  Mat Imagen( renglones, columnas, CV_64F, (unsigned char*) dummy.data() );
  const char *win0 = "Imagen ruidosa";      namedWindow( win0, WINDOW_AUTOSIZE );
  const char *win1 = "Estimaciones";          namedWindow( win1, WINDOW_AUTOSIZE );

  // genera patrones de franjas con datos sinteticos
  double valMax = -200.0, valMin = 200.0, num = 0.0, den = 0.0;
  double ro = double ( renglones ) / 2.0;        // calcula centro de la imagen
  double co = double ( columnas ) / 2.0;
  for ( int r = 0; r < renglones; r++ )
    for ( int c = 0; c < columnas; c++ )
      {

       if(opcion == 0)
        {      
        // compute posicion de la fase, valores [-dim,dim] 
        double dim = 2.5;
        double x = (double(r) - ro)*(dim/double(renglones/2.0));
        double y = (double(c) - co)*(dim/double(columnas/2.0));
            
        // calcula la fase 
        double phase = funcionPhase( x, y );
        valMax = (valMax > phase) ? valMax : phase;
        valMin =  (valMin < phase) ? valMin : phase;
        phase0(r,c) = phase;

        // genera patrones de franjas con ruido
        double ruido = factor*ruidoImagen.random();
        U_im0(r,c) = sin(phase+ruido); 
        U_re0(r,c) = cos(phase+ruido);
        WP0(r,c) = atan2(U_im0(r,c), U_re0(r,c));

        // fase envuelta sin ruido para determinar error
        WP(r,c) = atan2(sin(phase), cos(phase));
        
        // fase inicial con datos ruidosos
        U_im(r,c) = sin(WP0(r,c));
        U_re(r,c) = cos(WP0(r,c));

        // calcula el SNR 
        num += ( (phase+ruido) * (phase+ruido) );
        den += ( ruido * ruido );
        }
        else if(opcion == 1)
          {
          // Lee valores desde imagen
          double phase = double(IMAGEN.at<uchar>(r,c))/255.0;
          WP0(r,c) = 2.0 * M_PI * phase - M_PI;  
          
          // fase inicial con datos ruidosos
          U_im(r,c) = sin(WP0(r,c));
          U_re(r,c) = cos(WP0(r,c));
          }

      }

  if(opcion == 0)
  {
  // despliega SNR inicial
  cout << endl << "SNR = " << 20.0*log10(num/den) << " dB" << endl;
  dummy = (atan2( sin(phase0), cos(phase0) ) + M_PI) / (2.0*M_PI);  
  imshow( win0, Imagen );  
  
  // guarda imagenes iniciales
  dummy = (WP + M_PI) / (2.0*M_PI); 
  imwrite( "imagenes/phaseEnvuelta.pgm", 255*Imagen );
  dummy = (WP0 + M_PI) / (2.0*M_PI); 
  imwrite( "imagenes/phaseEnvueltaRuidosa.pgm", 255*Imagen );
  dummy = U_re0;
  normalize( Imagen, Imagen, 0, 1, NORM_MINMAX );
  imwrite( "imagenes/UreRuidosa.pgm", 255*Imagen );
  dummy = U_im0;
  normalize( Imagen, Imagen, 0, 1, NORM_MINMAX );
  imwrite( "imagenes/UimRuidosa.pgm", 255*Imagen );
  }
  if(opcion == 1)
  {
  cout << endl << "SNR = NA dB" << endl;
  dummy = (atan2( sin(WP0), cos(WP0) ) + M_PI) / (2.0*M_PI);  
  imshow( win0, Imagen );  
  
  
  // guarda imagenes iniciales
  dummy = (WP0 + M_PI) / (2.0*M_PI); 
  imwrite( "imagenes/phaseEnvueltaRuidosa.pgm", 255*Imagen );
  dummy = U_re0;
  normalize( Imagen, Imagen, 0, 1, NORM_MINMAX );
  imwrite( "imagenes/UreRuidosa.pgm", 255*Imagen );
  dummy = U_im0;
  normalize( Imagen, Imagen, 0, 1, NORM_MINMAX );
  imwrite( "imagenes/UimRuidosa.pgm", 255*Imagen );
  }


  // ************************************************************************
  //             Inicia procesamiento
  // ************************************************************************
  struct timeval start, end;      // variables de manejo del tiempo
  gettimeofday( &start, NULL );    // marca tiempo de inicio 

  // variables del metodo
  double epsilon1 = EPSILON1;    // criterio de paro del algoritmo
  double epsilon2 = EPSILON2;    // criterio de paro del algoritmo
  unsigned ITER1 = ITER_MAX1;    // maximo de iteraciones 
  unsigned iter = 0;             // contador de iteraciones   
  bool flag = true;

  double error;

  // inicia iteracion del algoritmo
  iter = 0;

  double Fx0 = Funcional( WP0, U_im, U_re );

  while ( flag )
    {
      
      // resguarda para calculo de error
      U_imo = U_im;
      U_reo = U_re;

      // calcula iteracion de Gauss-Seidel
      // retorna solucion actualizada en Is, Ic
      iteracion_Gauss_Seidel( WP0, U_im, U_re, U_im1, U_re1 );

      P = atan2(U_im, U_re);

      double Fx = Funcional( WP0, U_im, U_re );
      double difF = fabs(Fx0-Fx);    

      Fx0 = Fx;
      
      // calcula error de la estimación, despliega avances
      double errU_im = sqrt( sum( pow2(U_im-U_imo) ) ) / sqrt(sum( pow2(U_imo)));
      double errU_re = sqrt( sum( pow2(U_re-U_reo) ) ) / sqrt(sum( pow2(U_reo)));
      if ( (iter % 50) == 0 )
        {
          cout << "iteracion : " << iter << " Fx= " << Fx << " ||U_im||= " << errU_im << " ||U_re||= " << errU_re << endl;

          dummy = (atan2( U_im, U_re ) + M_PI) / (2.0*M_PI);
          imshow( win1, Imagen );        waitKey( 1 );
        }
        
      // criterios de paro || (difF < epsilon1)
      if ( (iter >= ITER1)  || (errU_im < epsilon1) || (errU_re < epsilon2))
        {
          cout << "iteracion : " << iter << " Fx = " << Fx << " ||U_im||= " << errU_im << " ||U_re||= " << errU_re << endl;
          flag = false;
        }

      // incrementa contador iteracion
      iter++;
    }

  // termina funcion, calcula y despliega valores indicadores del proceso  
  gettimeofday( &end, NULL );    // marca de fin de tiempo cronometrado   

  // ************************************************************************
  //   resultados del procesamiento
  // ************************************************************************

  // calcula tiempo utilizado milisegundos
  double startms = double(start.tv_sec)*1000. + double(start.tv_usec)/1000.;
  double endms = double(end.tv_sec)*1000. + double(end.tv_usec)/1000.;
  double ms = endms - startms;
  cout << endl << "Tiempo empleado  : " << ms << " mili-segundos" << endl; 

  // despliega diferencia entre la estimacion y el valor real
  P += (WP((renglones/2),(columnas/2)) - P((renglones/2),(columnas/2)));
  Po = fabs(P-WP); 
  error = sum( pow2(Po) ) / (double(renglones)*double(columnas));
  cout << endl << "RMS error : = " << sqrt( error ) << endl;
  error = sqrt( sum(pow2(Po)) ) / (sqrt( sum(pow2(P)) ) + sqrt( sum(pow2(WP)) ));
  cout << "Normalized error : = " << error << endl << endl;
  dummy = log( 1.0 + fabs(P - WP) );
  normalize( Imagen, Imagen, 0, 1, NORM_MINMAX );
  imshow( win1, Imagen );        //waitKey( 0 );
   
  // despliega estimacion final
  dummy = phase0;
  normalize( Imagen, Imagen, 0, 1, NORM_MINMAX );
  imshow( win0, Imagen );        
  dummy = P;
  normalize( Imagen, Imagen, 0, 1, NORM_MINMAX );
  imshow( win1, Imagen );        
  //waitKey( 0 );
  
  // guarda en archivo la estimacion
  dummy = U_im;
  imwrite( "imagenes/UimEstimada.pgm", 255*Imagen );
  dummy = U_re;
  imwrite( "imagenes/UreEstimada.pgm", 255*Imagen );
  dummy = (atan2( U_im, U_re ) + M_PI) / (2.0*M_PI);
  imwrite( "imagenes/phaseEstimada.pgm", 255*Imagen );


  // muestra phase en 3D
  FILE* gnuplot_pipe = popen( "gnuplot -p", "w" );
  Print3D( Po, gnuplot_pipe, "imagenes/phaseError.eps" );  
  if(opcion == 0)
  Print3D( phase0, gnuplot_pipe, "imagenes/phaseOriginal.eps" );
  
  pclose( gnuplot_pipe );  

  // termina ejecucion del programa
  return 0;
}


//*************************************************************************
//
//    Funciones de trabajo
//
//*************************************************************************
//*************************************************************************
//      genera valor de la fase para una posicion
//*************************************************************************
double funcionPhase( double x, double y )
{
  // declara variables de computo
  int tipo = TIPO;
  double phase;
  
  // selecciona termino de fase a utilizar
  switch ( tipo )
    {
      case 0:      // fase simple
        phase = 0.25*M_PI*( x*x + y*y ) / 0.2;
        break;
      case 1:      // fase tomada de la funcion peaks de Matlab
        phase = 5.0*( 3.0*(1.0-x)*(1.0-x)*exp(-x*x - (y+1.0)*(y+1.0)) - 10.0*((x/5.0) - x*x*x - y*y*y*y*y)
                  * exp(-x*x-y*y) - (1.0/3.0)*exp(-(x+1.0)*(x+1.0) - y*y) );
        break;
      case 2:      // fase tomada de la  ec. (24), AppOpt 51, p. 1257
//          x += 0.2;      y += 0.2;
          phase = 0.5*( 2.6 - 3.9*x 
                    - 2.6*( 1.0 - 6.0*y*y - 6.0*x*x + 6.0*y*y*y*y + 12.0*x*x*y*y + 6.0*x*x*x*x )
                    + 6.93*(5.0*x*y*y*y*y - 10.0*x*x*x*y*y + x*x*x*x*x )
                    + 0.86*(3.0*x - 12.0*x*y*y -12.0*x*x*x + 10.0*x*y*y*y*y + 20.0*x*x*x*y*y + 10.0*x*x*x*x*x )
                    + 5.2*(-4.0*y*y*y + 12.0*x*x*y + 5.0*y*y*y*y*y - 10.0*x*x*y*y*y - 15.0*x*x*x*x*y ) );
        break;
      case 3:      // fase tomada de la ec. (31), JOSA A 16, p. 475
        phase = 5.0 - 5.0*( 1.0 - 6.0*y*y - 6.0*x*x + 6.0*y*y*y*y + 12.0*x*x*y*y + 6.0*x*x*x*x )
                  + ( 3.0*x - 12.0*x*y*y - 12.0*x*x*x + 10.0*x*y*y*y*y + 20.0*x*x*x*y*y + 10.0*x*x*x*x*x );
        break;
      case 4:      // fase tomada de la ec. (9), OptLett 22, p. 1669
        phase = 3.0 - 4.5*x - 3.0*( 1.0 - 6.0*y*y - 6.0*x*x + 6.0*y*y*y*y + 12.0*x*x*y*y + 6.0*x*x*x*x )
                  + 8.0*(5.0*x*y*y*y*y - 10.0*x*x*x*y*y + x*x*x*x*x )
                  + (3.0*x - 12.0*x*y*y -12.0*x*x*x + 10.0*x*y*y*y*y + 20.0*x*x*x*y*y + 10.0*x*x*x*x*x )
                  + 6.0*(-4.0*y*y*y + 12.0*x*x*y + 5.0*y*y*y*y*y - 10.0*x*x*y*y*y - 15.0*x*x*x*x*y ); 
        break;
      case 5:      // fase tomada de la  ec. (30), AppOpt 49, p. 6224
          x *= 2.0;       y*= 2.0;
        phase = 2.0*x*y + 4.0*(2.0*(x*x + y*y) - 1.0) + 2.0*(3.0*(x*x + y*y)*y - 2.0*y);
        break;
      case 6:      // fase tomada de la ec. (13),  Proc. SPIE articulo 849319
        x -= 0.5;      
        phase = 45.0*( 2.0*x*y + 8.0*(x*x+y*y) + 6.0*(x*x*x*x+y*y*y*y)*y - 4.0*(y+1) );
        break;
      case 7:      // fase tomada de la ec. (12),  Proc. SPIE articulo 849319
        x -= 0.1;      y -= 0.05;
        phase = 5.0*( 30.0*(x*x + y*y - x*x*x*x - y*y*y*y) - 60.0*x*x*y*y + 3.0*x - 12.0*(x*y*y + x*x*x) 
                 + 10.0*x*y*y*y*y + 20*x*x*x*y*y + 10*x*x*x*x*x );
        break;
      case 8:      // fase tomada de la ec. (11),  Proc. SPIE articulo 849319
        x += 0.2;      y += 0.2;
        phase = 13.75*( -1.5*x + 18.0*(x*x + y*y) - 12.0*y*y*y*y - 36.0*x*x*y*y - 18.0*x*x*x*x
                 + 50.0*x*y*y*y*y - 60.0*x*x*x*y*y + 8.0*x*x*x*x*x -12.0*(x*y*y + x*x*x)
                 + 10.0*x*x*x*x*x - 24.0*y*y*y + 72.0*x*x*y + 30.0*y*y*y*y*y - 60.0*x*x*y*y*y
                 + 90.0*x*x*x*x*y );
        break;

      case 9:      // plano
        phase = 4.0*M_PI*x;
        if ( y < 0.0 )    phase = -1.0*phase - 0.0;
        break;

      case 10:      // plano
        phase = 2.0*M_PI*x;
        break;

      case 11:      // fase tomada de 
        phase = 2.0*2.0*M_PI*( x*x + y*y ) / 0.2;
        if ( x < 0.0 )    phase *= -1.0;
        break;
    }

  // regresa fase
  return phase; 
}


// ************************************************************************
//       funcion principal de la derivada
//*************************************************************************
void Derivada( Array<double,2>& dIs, Array<double,2>& dIc, Array<double,2>& WP, Array<double,2>& Is, Array<double,2>& Ic )
{
  // define variables a a utilizar
  // parametro de regularizacion
  double lambda1 = LAMBDA1, lambda2 = LAMBDA2, lambda3 = LAMBDA3;      
  double beta = BETA;
  int columnas = WP.cols();
  int renglones = WP.rows();
  double Ux, Uy, divIs, divIc;
  double V31xIs, V32xIs, V31yIs, V32yIs;
  double V31xIc, V32xIc, V31yIc, V32yIc;
  double U, V, auxU, auxV, v1, v2;

  // evalua primera derivada de fase y
  // terminos de divergencia
  for ( int r = 0; r < renglones; r++ )
    for ( int c = 0; c < columnas; c++ )
      {
       // termino para divergencia de nabla phi / |nabla phi|
        // procesa condiciones de frontera, eje x
        if ( r == renglones-1 || c == 0 || c == columnas-1 ) {  V31xIs = 0.0; V31xIc = 0.0; }
        else
          {
            Ux = Is(r+1,c) - Is(r,c);
            Uy = minMod( 0.5*(Is(r+1,c+1) - Is(r+1,c-1)), 0.5*(Is(r,c+1) - Is(r,c-1)) );
            V31xIs = Ux / sqrt( Ux*Ux + Uy*Uy + beta );
            Ux = Ic(r+1,c) - Ic(r,c);
            Uy = minMod( 0.5*(Ic(r+1,c+1) - Ic(r+1,c-1)), 0.5*(Ic(r,c+1) - Ic(r,c-1)) );
            V31xIc = Ux / sqrt( Ux*Ux + Uy*Uy + beta );
          }  

        if ( r == 0 || c == 0 || c == columnas-1 ) {  V32xIs = 0.0; V32xIc = 0.0; }
        else
          {           
            Ux = Is(r,c) - Is(r-1,c);
            Uy = minMod( 0.5*(Is(r,c+1) - Is(r,c-1)), 0.5*(Is(r-1,c+1) - Is(r-1,c-1)) );
            V32xIs = Ux / sqrt( Ux*Ux + Uy*Uy + beta );
            Ux = Ic(r,c) - Ic(r-1,c);
            Uy = minMod( 0.5*(Ic(r,c+1) - Ic(r,c-1)), 0.5*(Ic(r-1,c+1) - Ic(r-1,c-1)) );
            V32xIc = Ux / sqrt( Ux*Ux + Uy*Uy + beta );
          }  
      
        // procesa condiciones de frontera, eje y
        if ( c == columnas-1 || r == 0 || r == renglones-1) { V31yIs = 0.0; V31yIc = 0.0; }
        else
          {            
            Ux = minMod( 0.5*(Is(r+1,c+1) - Is(r-1,c+1)), 0.5*(Is(r+1,c) - Is(r-1,c)) );
            Uy = Is(r,c+1) - Is(r,c);
            V31yIs = Uy / sqrt( Ux*Ux + Uy*Uy + beta );
            Ux = minMod( 0.5*(Ic(r+1,c+1) - Ic(r-1,c+1)), 0.5*(Ic(r+1,c) - Ic(r-1,c)) );
            Uy = Ic(r,c+1) - Ic(r,c);
            V31yIc = Uy / sqrt( Ux*Ux + Uy*Uy + beta );
          }  

        if ( c == 0 || r == 0 || r == renglones-1) {  V32yIs = 0.0; V32yIc = 0.0; }
        else
          {
            Ux = minMod( 0.5*(Is(r+1,c) - Is(r-1,c)), 0.5*(Is(r+1,c-1) - Is(r-1,c-1)) );
            Uy = Is(r,c) - Is(r,c-1);
            V32yIs = Uy / sqrt( Ux*Ux + Uy*Uy + beta );
            Ux = minMod( 0.5*(Ic(r+1,c) - Ic(r-1,c)), 0.5*(Ic(r+1,c-1) - Ic(r-1,c-1)) );
            Uy = Ic(r,c) - Ic(r,c-1);
            V32yIc = Uy / sqrt( Ux*Ux + Uy*Uy + beta );
          }

       divIs = (V31xIs-V32xIs) + (V31yIs-V32yIs);
       divIc = (V31xIc-V32xIc) + (V31yIc-V32yIc);

//       dIs(r,c) = dIs(r,c) - divIs;
//       dIc(r,c) = dIc(r,c) - divIc;
       dIs(r,c) = lambda2 * (Is(r,c) - sin(WP(r,c))) + 2.0*lambda3*Is(r,c)*(Is(r,c)*Is(r,c) + Ic(r,c)*Ic(r,c) - 1.0) - divIs;
       dIc(r,c) = lambda1 * (Ic(r,c) - cos(WP(r,c))) + 2.0*lambda3*Ic(r,c)*(Is(r,c)*Is(r,c) + Ic(r,c)*Ic(r,c) - 1.0) - divIc;
      } 
  
}
// ************************************************************************
//       funcion para Gauss-Seidel
//*************************************************************************
void iteracion_Gauss_Seidel( Array<double,2>& WP, Array<double,2>& Is, Array<double,2>& Ic, Array<double,2>& Is1, Array<double,2>& Ic1 )
{
  // define parametro de regularizacion
  int columnas = WP.cols();
  int renglones = WP.rows();
  double lambda1 = LAMBDA1, lambda2 = LAMBDA2, lambda3 = LAMBDA3;
  double beta = BETA;
  double V1x, V2x, V1y, V2y, Ux, Uy;
  double AIs, BIs, CIs, DIs, numIm, denIm;
  double AIc, BIc, CIc, DIc, numReal, denReal;
  double auxIm, auxReal;
  
  //condiciones de frontera Neumann para Is, Ic
  boundaryCond1( Is );
  boundaryCond1( Ic );
  Is1 = Is;
  Ic1 = Ic;
  // calculo de punto fijo K iteraciones de GS
  for ( int k = 0; k < K; k++ )
  {
  for ( int r = 1; r < renglones-1; r++ )
   {
    for ( int c = 1; c < columnas-1; c++ )
      {
        // procesa condiciones de frontera, eje x
        if ( r == renglones-1 )
          { AIs = 0.0; AIc = 0.0; }
        else
          {
            Ux = Is(r+1,c) - Is(r,c);
            Uy = Is(r,c+1) - Is(r,c);
            AIs = 1.0 / sqrt( Ux*Ux + Uy*Uy + beta );
            Ux = Ic(r+1,c) - Ic(r,c);
            Uy = Ic(r,c+1) - Ic(r,c);
            AIc = 1.0 / sqrt( Ux*Ux + Uy*Uy + beta );
          }  
        if ( r == 0 )
          { BIs = 0.0; BIc = 0.0; }
        else
          {         
            Ux = Is(r,c) - Is(r-1,c);
            Uy = Is(r-1,c+1) - Is(r-1,c);
            BIs = 1.0 / sqrt( Ux*Ux + Uy*Uy + beta );
            Ux = Ic(r,c) - Ic(r-1,c);
            Uy = Ic(r-1,c+1) - Ic(r-1,c);
            BIc = 1.0 / sqrt( Ux*Ux + Uy*Uy + beta );
          }  
      
        // procesa condiciones de frontera, eje y
        if ( c == columnas-1 )
          { CIs = 0.0; CIc = 0.0; }
        else
          {           
            Ux = Is(r+1,c) - Is(r,c);
            Uy = Is(r,c+1) - Is(r,c);
            CIs = 1.0 / sqrt( Ux*Ux + Uy*Uy + beta );
            Ux = Ic(r+1,c) - Ic(r,c);
            Uy = Ic(r,c+1) - Ic(r,c);
            CIc = 1.0 / sqrt( Ux*Ux + Uy*Uy + beta );
          }  
        if ( c == 0 )
          {  DIs = 0.0; DIc = 0.0;} 
        else
          {
            Ux = Is(r+1,c-1) - Is(r,c-1);
            Uy = Is(r,c) - Is(r,c-1);
            DIs = 1.0 / sqrt( Ux*Ux + Uy*Uy + beta );
            Ux = Ic(r+1,c-1) - Ic(r,c-1);
            Uy = Ic(r,c) - Ic(r,c-1);
            DIc = 1.0 / sqrt( Ux*Ux + Uy*Uy + beta );
          }

        // iteracion de Gauss-Seidel
        numReal = lambda1*(cos(WP(r,c)) + 2.0*lambda3*Ic(r,c)) + AIc*Ic(r+1,c) + BIc*Ic1(r-1,c) + CIc*Ic(r,c+1) + DIc*Ic1(r,c-1);
        numIm = lambda2*(sin(WP(r,c)) + 2.0*lambda3*Is(r,c)) + AIs*Is(r+1,c) + BIs*Is1(r-1,c) + CIs*Is(r,c+1) + DIs*Is1(r,c-1);

        auxReal = 2.0*lambda3*( Is(r,c)*Is(r,c) + Ic(r,c)*Ic(r,c) ) + lambda1;
        auxIm = 2.0*lambda3*( Is(r,c)*Is(r,c) + Ic(r,c)*Ic(r,c) ) + lambda2;
        
        denIm = auxIm + (AIs + BIs + CIs + DIs);
        denReal = auxReal + (AIc + BIc + CIc + DIc);
        Is1(r,c) = numIm / denIm;
        Ic1(r,c) = numReal / denReal;
      } // termina ciclo c
//     Is = Is1;
//     Ic = Ic1;
    } // termina ciclo r
     Is = Is1;
     Ic = Ic1;
   } // termina ciclo k
}


// ***************************************************************
//   min-mod
// ***************************************************************
double minMod( double a, double b )
{
  // minmod operator
  double signa = (a > 0.0) ? 1.0 : ((a < 0.0) ? -1.0 : 0.0);
  double signb = (b > 0.0) ? 1.0 : ((b < 0.0) ? -1.0 : 0.0);
//  double minim = fmin( fabs(a), fabs(b) ); 
  double minim = ( fabs(a) <= fabs(b) ) ? fabs(a) : fabs(b); 
  return ( (signa+signb)*minim/2.0 );

  // geometric average
//  return( 0.5*(a+b) ); Total Variation Diminishing Runge-Kutta Schemes
  
  // upwind 
//  double maxa = (a > 0.0) ? a : 0.0;
//  double maxb = (b > 0.0) ? b : 0.0;
//  return( 0.5*(maxa+maxb) );  
}
// ************************************************************************
//       funcional
//*************************************************************************
double Funcional( Array<double,2>& WP, Array<double,2>& Is, Array<double,2>& Ic )
{
  // define parametro de regularizacion
  int columnas = WP.cols();
  int renglones = WP.rows();
  double lambda1 = LAMBDA1, lambda2 = LAMBDA2, lambda3 = LAMBDA3, val = 0.0, v0, v1, v2, v3, a1, a2, a3;
  double dxIs, dyIs, dxIc, dyIc;
  double hx = 1.0 / (double(renglones)-1.0);
  double hy = 1.0 / (double(columnas)-1.0);

  // evalua derivadas parciales para funcional
  for ( int r = 0; r < renglones; r++ )
    for ( int c = 0; c < columnas; c++ )
      {
        // campo de gradiente de la informacion
        if ( c == 0 )
          {  dyIs = Is(r,c+1) - Is(r,c);
             dyIc = Ic(r,c+1) - Ic(r,c); }
        else if ( c == columnas-1 )
          {  dyIs = Is(r,c)-Is(r,c-1);
             dyIc = Ic(r,c)-Ic(r,c-1); }
        else
          {  dyIs = Is(r,c+1) - Is(r,c);//0.5*(Is(r,c+1)-Is(r,c-1)); 
             dyIc = Ic(r,c+1) - Ic(r,c);}//0.5*(Ic(r,c+1)-Ic(r,c-1)); }
          
        // campo de gradiente de la informacion
        if ( r == 0 )
          {  dxIs = Is(r+1,c)-Is(r,c);
             dxIc = Ic(r+1,c)-Ic(r,c); }
        else if ( r == renglones-1 )
          {  dxIs = Is(r,c)-Is(r-1,c);
             dxIc = Ic(r,c)-Ic(r-1,c);}
        else
          {  dxIs = Is(r+1,c)-Is(r,c);//0.5*(Is(r+1,c)-Is(r-1,c)); 
             dxIc = Ic(r+1,c)-Ic(r,c);}//0.5*(Ic(r+1,c)-Ic(r-1,c)); }
       // termina calculo de derivadas parciales de fase

       a1 = Ic(r,c)-cos(WP(r,c));
       a2 = Is(r,c)-sin(WP(r,c));
       a3 = Is(r,c)*Is(r,c) + Ic(r,c)*Ic(r,c) - 1.0;
       v0 = 0.5 * lambda1 * a1*a1 + 0.5 * lambda2 * a2*a2;
       v1 = 0.5 * lambda3 * a3 * a3;
       v2 = sqrt(dxIs*dxIs + dyIs*dyIs);
       v3 = sqrt(dxIc*dxIc + dyIc*dyIc);
       val += v0 + v1 + v2 + v3;
      }

  // regresa valor
  return val * hx * hy;
}

//*************************************************************************
//      obtiene las diferencias envueltas del termino de fase
//*************************************************************************
double gradientWrap( double p1, double p2 )
{
  double r = p1 - p2;
  return atan2( sin(r), cos(r) ); 
}

// ***************************************************************
//   Condiciones de frontera Neumann
// ***************************************************************
void boundaryCond1( Array<double,2>& T )
{
  // define parametro de regularazacion
  int columnas = T.cols();
  int renglones = T.rows();
  blitz::Range all = blitz::Range::all();

  // condiciones de frontera
  T(0,all) = T(1,all);
  T(renglones-1,all) = T(renglones-2,all);
  T(all,0) = T(all,1);
  T(all,columnas-1) = T(all,columnas-2);
  T(0,0) = T(1,1);
  T(0,columnas-1) = T(1,columnas-2);
  T(renglones-1,0) = T(renglones-2,1);
  T(renglones-1,columnas-1) = T(renglones-2,columnas-2);
}
void boundaryCond2( Array<double,2>& T )
{
  // define parametro de regularazacion
  int columnas = T.cols();
  int renglones = T.rows();
  blitz::Range all = blitz::Range::all();

  // condiciones de frontera
  T(0,all) = T(1,all) = T(2,all);
  T(renglones-1,all) = T(renglones-2,all) = T(renglones-3,all);
  T(all,0) = T(all,1) = T(all,2);
  T(all,columnas-1) = T(all,columnas-2) = T(all,columnas-3);
  T(0,0) = T(1,1) = T(2,2);
  T(0,columnas-1) = T(1,columnas-2) = T(2,columnas-3);
  T(renglones-1,0) = T(renglones-2,1) = T(renglones-3,2);
  T(renglones-1,columnas-1) = T(renglones-2,columnas-2) = T(renglones-3,columnas-3);
}


//*************************************************************************
//
//    Funciones para despliegue de resultados
//
//*************************************************************************
void Print3D( Array<double,2> Z, FILE* salida, const char* fileName )
{
  // salida en modo grafico o en archivo
  fprintf( salida, "set terminal postscript eps enhanced rounded\n");
  fprintf( salida, "set output \"%s\"\n", fileName );  
  fprintf( salida, "set style line 1 linetype -1 linewidth 1\n" );
  fprintf( salida, "set xlabel \"columns (pixels)\" offset -1,-1\n" );
  fprintf( salida, "set ylabel \"rows (pixels)\" offset -1,-1\n" );
  fprintf( salida, "set zlabel \"phase\"\n" );
  fprintf( salida, "set xrange [%f:%f]\n", 0., float(Z.cols()) );
  fprintf( salida, "set yrange [%f:%f]\n", 0., float(Z.rows()) );
  //fprintf( salida, "set zrange [%f:%f]\n", -100.0, 100.0 );
  fprintf( salida, "set xtics 100 offset -0.5,-0.5\n" );
  fprintf( salida, "set ytics 100 offset -0.5,-0.5\n" );
  fprintf( salida, "set view 70, 210\n" );
  fprintf( salida, "unset key\n" );
  fprintf( salida, "unset colorbox\n" );
  fprintf( salida, "set hidden3d front\n" );
//  fprintf( salida, "set xyplane at -45.0\n" );
  fprintf( salida, "splot '-' using 1:2:3 title '' with lines lt -1 lw 0.1\n" );
  for ( int c = 0; c < Z.cols(); c += 8 )
    {
      for ( int r = 0; r < Z.rows(); r += 8 )
        fprintf( salida, "%f %f %f\n", float(c), float(r), float(Z(r,c)) );
      fprintf(  salida, "\n" );      // New row (datablock) separated by blank record
    }
  fprintf( salida, "e\n" );
  fflush( salida );
}
//*************************************************************************
//
//    Funciones para calculo de gradiente
//
//*************************************************************************
void gradienteArray(Array<double,2>& P, Array<double,2>& Px, Array<double,2>& Py)
{
  int columnas = P.cols();
  int renglones = P.rows();
  for ( int r = 0; r < renglones; r++ )
    for ( int c = 0; c < columnas; c++ )
      {
        // campo de gradiente de la informacion
        if ( c == 0 )
          Px(r,c) = P(r,c+1)-P(r,c);
        else if ( c == columnas-1 )
          Px(r,c) = P(r,c)-P(r,c-1);
        else
          Px(r,c) = 0.5*(P(r,c+1)-P(r,c-1));
          
        // campo de gradiente de la informacion
        if ( r == 0 )
          Py(r,c) = P(r+1,c)-P(r,c);
        else if ( r == renglones-1 )
          Py(r,c) = P(r,c)-P(r-1,c);
        else
          Py(r,c) = 0.5*(P(r+1,c)-P(r-1,c));
      }
}


