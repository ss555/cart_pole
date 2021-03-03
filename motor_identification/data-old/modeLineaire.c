#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pigpio.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <sys/types.h>
#include <termios.h>

#include "rotary_encoder.h"

#define PI 3.1415
#define SLEEP 50000   //50000 = 50ms 
#define NB_ENABLE  1
#define NB_DISABLE 0
#define TAILLE_MAX 25000
#define KALMAN 0 


/*
 *
 * Ce programe est la premiere version du pendule inverse (mode linearise) 
 * Pour le cablage :
 * encodeur 1 (position du chariot) : cable vert => GPIO21 et cable blanc => GPIO20)
 * encodeur 2 (angle du pendule) : cable vert => GPIO19 et cable blanc => GPIO26)
 * moteur (signal PWM) : fil jaune => GPIO24
 * moteur (signal DIR) : fil vert => GPIO16
 * 
 * Pour compiler :
 * 	gcc -o modeLineaire modeLineaire.c rotary_encoder.c -lpigpio -lrt
 *
 * Pour executer :
 * 	sudo ./modeLineaire
 */


//Pour pouvoir soir sur appui touche
int kbhit(void);
void nonblock(int);


void callback_moteur(int);
void callback_pendule(int);


// Declaration et init des variables
// Pour le chariot
int ipuls_chariot = 0;
int ipuls_chariot_old = 0;
int isens_chariot = 0;
int isens_chariot_old = 0;
double fpos_chariot = 0.;
double fpos_chariot_old= 0.;
double fvit_chariot = 0.;

// Pour le pendule
int ipuls_pendule = 0;
int ipuls_pendule_old = 0;
int isens_pendule = 0;
double fangle_pendule = PI;
double fangle_pendule_old = PI;
double frot_pendule = 0.;

double etatRecons[4]={0.,0.,PI,0.};

double etatMeasure[2]={0,PI};
double etatMeasureOld[2]={0,PI};

// Pour le controle, ampliture et direction
int icontrol;
int idirection;
double fcontrol;

//Pondération pour le balancement
double kswing = 50  ;
//Pondération pour le déplacement du chariot (consigne x = 0)
double ksx = 2.5;
double ftemp, fsign;

// Pour la sécurité
int iFinCourseEncodeur;
int iFinCourseMoteur;
int iSortieBute = 0;


int main(int argc, char *argv[])
{
	//Pour le debug
	FILE *fichier;
	double debug[8][TAILLE_MAX]={0};
	int k;
	
	//Pour la sortie
	int fin = 0 ;
	char carac;
	nonblock(NB_ENABLE);

	//variables diverses
	long mnloops = 0;
	double mttot = 0.0;
	int itmp = 0;
	double Te = (double)SLEEP*1e-6;	
	
	//Variables moteur (sign de fC pris en compte dans l'equation)
	float fA = 14.3, fB = 0.76, fC = 0.44;

	//Varaibles gain
	//ok
	//float Kx = -8, Kv = -36, Kth = 77, Kw = 12;
	
	//ORIGINAL par matlab float Kx = -10, Kv = -42, Kth = 103, Kw = 23;	
	//float Kx = -9, Kv = -40, Kth = 85, Kw = 15;	
	//ok si on ajoute la ligne qui converti F en U (ce qui ne doit pas etre le cas normalement)
	//float Kx = -25, Kv = -58, Kth = 75, Kw = 12;
      
	///////////////////////////////////////////////
	//float Kx = -22, Kv = -42, Kth = 88, Kw = 16.5;
	///////////////////////////////////////////////
	float Kx = -23, Kv = -44, Kth = 100, Kw = 17;

	////DEB KALMAN////
	double Ad[16] = {1.e+00,2.40980670e-02,3.10165022e-04,-1.94276863e-05,
 				     0.0e+00,1.84438276e-01,9.56421512e-03,-4.69791358e-04,
 				     0.0e+00,-7.48695997e-02,1.03647093e+00,4.76402646e-02,
 				    -0.0e+00,-2.30867089e+00,1.43074563e+00,9.19794428e-01};
 	double Bd[4] = {0.0013301,0.0418802,0.00384466,0.11855337};
 	double Cd[8] = {1., 0., 0., 0., 0., 0., 1., 0.};

 	double PKalman[16] = {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
 	double Rx = 0.1, Rth = 0.1;
 	double R_k[4] = {Rx*Rx, 0, 0, Rth*Rth};
 	double Qx = 0.01,Qxp = 0.01, Qth = 0.01, Qthp = 0.01;
	double Q_k[16] = {Qx, 0, 0, 0, 0, Qxp, 0, 0, 0, 0, Qth, 0, 0, 0, 0, Qthp};
	////FIN KALMAN////


   	//Declaration de l'encodeur chariot et pendule
   	Pi_Renc_t *enc_moteur;
   	Pi_Renc_t *enc_pendule;

   	//Init du GPIO, sortie si erreur
   	if (gpioInitialise() < 0) 
   	{
		printf("Pb init gpioInit\n");
		return 1;
   	}
   	printf("GPIO OK\n");

	//direction moteur
	gpioSetMode(16, PI_OUTPUT); 
	//Fin de course
	gpioSetMode(17, PI_INPUT); //FDC17 
	gpioSetMode(18, PI_INPUT); //FDC18
	gpioSetPullUpDown(17, PI_PUD_UP);   
	gpioSetPullUpDown(18, PI_PUD_UP);   

   	enc_moteur = Pi_Renc(20, 21, callback_moteur);
	enc_pendule = Pi_Renc(19,26, callback_pendule);

	iFinCourseMoteur = gpioRead(17);	//1 si pas appui
	iFinCourseEncodeur = gpioRead(18);  //1 si pas appui

	//Deb initialisation position
	puts("!! INITIALISATION DE LA POSITION DU CHARIOT !!");
	//Deb initialisation de la position (CAS 1 : aucun appui sur FDC)
	if (iFinCourseEncodeur==1 && iFinCourseMoteur==1)
	{
		while(iFinCourseEncodeur==1)
		{
			// On definit le sens (vers encodeur => 1)
			gpioWrite(16,1);
			// On envoie au moteur (50 bonne vitesse)
			gpioPWM(24,50);
			// Mesure fin de course 
			iFinCourseEncodeur = gpioRead(18);
			//dodo
			usleep(10000);
		}
		gpioPWM(24,0);
		sleep(1);
		// On est en butée encodeur.
		// Raz des pulses et go au centre (42.cm en direction 0)
		fpos_chariot = 0;
		ipuls_chariot = 0;
		while(fpos_chariot >= -0.42)
		{
			//On definit le sens (vers encodeur => 0)
			gpioWrite(16,0);
			//On envoie au moteur (50 bonne vitesse)
			gpioPWM(24,50);
			//Nouvelle position du chariot
			fpos_chariot = ((ipuls_chariot * 2 * PI) / 600 )* 0.006371;
			//dodo
			usleep(10000);
		}
		//On est au centre (on met tout à zéro)
		gpioPWM(24,0);
		fpos_chariot = 0.;
		ipuls_chariot = 0;
	}
	// Fin initialisation de la position CAS1

	//Deb initialisation de la position (CAS 2 : appui sur FDC encodeur)
	else if (iFinCourseEncodeur==0)
	{
		// On est en butée encodeur.
		// Raz des pulses et go au centre (42.cm en direction 0)
		fpos_chariot = 0;
		ipuls_chariot = 0;
		while(fpos_chariot >= -0.42)
		{
			//On definit le sens (vers encodeur => 0)
			gpioWrite(16,0);
			//On envoie au moteur (50 bonne vitesse)
			gpioPWM(24,50);
			//Nouvelle position du chariot
			fpos_chariot = ((ipuls_chariot * 2 * PI) / 600 )* 0.006371;
			//dodo
			usleep(10000);
		}
		//On est au centre (on met tout à zéro)
		gpioPWM(24,0);
		fpos_chariot = 0.;
		ipuls_chariot = 0;
	}
	// Fin initialisation de la position CAS2

	//Deb initialisation de la position (CAS 3 : appui sur FDC moteur)
	else if (iFinCourseMoteur==0)
	{
		// On est en butée encodeur.
		// Raz des pulses et go au centre (42.cm en direction 0)
		fpos_chariot = 0;
		ipuls_chariot = 0;
		while(fpos_chariot <= 0.42)
		{
			//On definit le sens (vers encodeur => 0)
			gpioWrite(16,1);
			//On envoie au moteur (50 bonne vitesse)
			gpioPWM(24,50);
			//Nouvelle position du chariot
			fpos_chariot = ((ipuls_chariot * 2 * PI) / 600 )* 0.006371;
			//dodo
			usleep(10000);
		}
		//On est au centre (on met tout à zéro)
		gpioPWM(24,0);
		fpos_chariot = 0.;
		ipuls_chariot = 0;
	}
	puts("!! FIN INITIALISATION DE LA POSITION DU CHARIOT !!");
	// Fin initialisation de la position CAS3
	// Fin initialisation position


	///////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////
	// Début du mode lineaire
	///////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////


	iFinCourseMoteur = gpioRead(17);	//1 si pas appui
	iFinCourseEncodeur = gpioRead(18);  //1 si pas appui

	while(!fin && iFinCourseMoteur==1 && iFinCourseEncodeur ==1)
	{
    	if (itmp==0)
		{
			puts("!! MODE LINEAIRE => ON !!");
		}

    	// Divers pour le temps de calcul (DEBUT)
		time_t mnow;
	    struct timeval mtv;
	    struct timezone mtz;
	    time(&mnow);
	    gettimeofday(&mtv, &mtz);
		double startTime = (double)mnow + mtv.tv_usec/1e6;


 		//Mesure de la position du chariot
		//n_step = 600, diametre de l'axe = 6.6mm
		etatMeasure[0] = ((ipuls_chariot * 2 * PI) / 600 )* 0.006371;

		//Mesure de l'angle du pendule
		//n_step = 600
		//fangle_pendule = ((2 * PI) * (ipuls_pendule % 600)) / (600.) - (PI);
		etatMeasure[1] = (PI + 2 * PI  * ipuls_pendule / 600.) ;
		while( etatMeasure[1] > PI)
		{
			etatMeasure[1] -= 2 * PI;
		}
		while( etatMeasure[1] < -PI)
		{
			etatMeasure[1] += 2 * PI;
		}
		
		////////////////////////////////
		//DEB RECONSTRUCTION DE L'ETAT//
		////////////////////////////////

		if (KALMAN == 0)
		{
			etatRecons[0] = etatMeasure[0];
			etatRecons[1] = (etatMeasure[0] - etatMeasureOld[0]) / Te;
			etatRecons[2] = etatMeasure[1];

			//OK Cas bizarre de la dérive dans le sens + vers - en position haute
			if (etatMeasureOld[1]>0 && etatMeasure[1]<0 && etatMeasure[1] > -PI/2)
			{
				etatRecons[3] = ( etatMeasure[1] - etatMeasureOld[1]) /Te;
				// printf("1) + vers - en PH au temps %g\n",mttot);
			}

			//NON Cas bizarre de la dérive dans le sens - vers + en position haute
			else if (etatMeasureOld[1]<0 && etatMeasure[1]>0 && etatMeasure[1] < PI/2)
			{
				etatRecons[3] = ( etatMeasure[1] - etatMeasureOld[1] ) /Te;
				// printf("2) - vers + en PH au temps %g\n",mttot);
			}
			
			//OK Cas bizarre de la dérive dans le sens + vers - en position basse
			else if (etatMeasureOld[1]>0 && etatMeasure[1]<0 && etatMeasureOld[1] > PI/2)
			{
				etatRecons[3] = ((PI-etatMeasureOld[1]) + (PI + etatMeasure[1]) ) /Te;
				// printf("3) + vers - en PB au temps %g\n",mttot);		
			}
			//NON Cas bizarre de la dérive dans le sens - vers + en position basse
			else if (etatMeasureOld[1]<0 && etatMeasure[1]>0 && etatMeasureOld[1] < -PI/2)
			{
				etatRecons[3] = ((-PI-etatMeasureOld[1]) - (PI - etatMeasure[1]) ) /Te;
				// printf("4) - vers + en PB au temps %g\n",mttot);				
			}

			//Cas normal
			else
				etatRecons[3] = (etatMeasure[1] - etatMeasureOld[1]) / Te;
		}
		else (KALMAN == 1)
		{
			
		}

	    	
     	//Si dans la partie linearisable (150 et 210 degrés)
     	if (etatRecons[2] >= -0.2 && etatRecons[2] <=0.2)
     	{

			//ou mettre FC????????
			fcontrol = Kx * etatRecons[0] + Kv * etatRecons[1] + Kth * etatRecons[2] + Kw * etatRecons[3]; //frot_pendule_filtre;
			fcontrol = fcontrol + copysign (fC, etatRecons[1]);

			//printf("Control1  fcontrol = %g \n",fcontrol);
			//fcontrol = (fcontrol + fA * fvit_chariot + copysignf (fC, fvit_chariot)) / fB;
			//printf("Control2  fcontrol = %g \n",fcontrol);
			fcontrol = 255. * fcontrol / 12.;
			//printf("Control3  fcontrol = %g \n",fcontrol);

			if (fcontrol > 0)
			{
				idirection = 0;
			}
			else if (fcontrol <0)
			{
				idirection = 1;
			}
			
			fcontrol = fabs(fcontrol);
			 
			if (fcontrol >= 255)
			{
				fcontrol = 255.;
			}
			else if (fcontrol < 0)
			{
				fcontrol = 0.;
			}
			
			icontrol = fcontrol;
			//printf("Control4 fcontrol = %g et icontrol = %d\n",fcontrol,icontrol);

		}
		else
		{
		  	fcontrol = 0;
		  	// fsign = copysignf( 1., frot_pendule*cos(fangle_pendule));
		  	// fcontrol = kswing*(1-cos(fangle_pendule)) * fsign ;//- ksx*fpos_chariot;
		  	// printf("Swing up en cours, fcontrol = %g",fcontrol);
		  	// printf("frot_pendule = %g , c(fangle_pendule) = %g",frot_pendule,cos(fangle_pendule));
		  	// printf("fsign=%g\n",fsign);

			// if (fcontrol > 0)
			// 	{
			// 		idirection = 0;
			// 	}
			// else if (fcontrol <0)
			// 	{
			// 		idirection = 1;
			// 	}
				
			// fcontrol = fabs(fcontrol);
				 
			// if (fcontrol >= 255)
			// 	{
			// 		fcontrol = 255.;
			// 	}
			// else if (fcontrol < 0)
			// 	{
			// 		fcontrol = 0.;
			// 	}	
			// icontrol = fcontrol;
		}

		//Controle de la position du chariot !
		// Marge de sécurité de 10cm
		if (etatRecons[0] > 0.35)	
		{
			puts("DANGER Niveau 1 COTE ENCODEUR (Zone des 35cm)");
			if(icontrol > 0 && idirection == 1)
			{
				puts("DANGER Niveau 2 COTE ENCODEUR, on arrete!");
				icontrol = 0;
				break;
			}
		}
		if (etatRecons[0] < -0.35)
		{
			puts("DANGER Niveau 1 COTE MOTEUR (Zone des 35cm)");
			if(icontrol > 0 && idirection == 0)
			{
				puts("DANGER Niveau 2 COTE MOTEUR, on arrete!");
				icontrol = 0;
				break;
			}
		}
		//Fin controle de la position du chariot !
		
		//On definit le sens
		gpioWrite(16,idirection);
		
		//On envoie au moteur
		gpioPWM(24,icontrol);	
     	
		//temps de sommeil en microseconde
		usleep(SLEEP);

		// Divers pour temps de calcul (FIN)
		time(&mnow);
		gettimeofday(&mtv, &mtz);
		Te = (((double)mnow + mtv.tv_usec/1e6) - startTime);
		if (Te > (SLEEP/1e6 + 0.15*SLEEP/1e6))
		{
			printf("Pb de temps, iter = %g au lieu de %d!!!!!!!!!!!\n",Te,SLEEP/1e6);
			//Te = SLEEP/1e6;
		}
		mttot += Te;
		mnloops++;


		// Sauvegarde pour debug
		debug[0][itmp] = (double)itmp;
		debug[1][itmp] = (double)mttot;
		debug[2][itmp] = (double)etatRecons[0];
     	debug[3][itmp] = (double)etatRecons[1];
     	debug[4][itmp] = (double)etatRecons[2];
     	debug[5][itmp] = (double)etatRecons[3];
     	debug[6][itmp] = (double)Te;
     	if (idirection==0)
     		debug[7][itmp] = -(double)icontrol;
     	else if (idirection==1)
     		debug[7][itmp] = (double)icontrol;


		//Sauvegarde des anciennes valeurs pour le calcul de la vitesse
		etatMeasureOld[0] = etatRecons[0];
		etatMeasureOld[1] = etatRecons[2];
		// Fin de la sauvegardes des anciennes valeurs
		

		// Deb mesure fin de course 
		iFinCourseMoteur = gpioRead(17);
		iFinCourseEncodeur = gpioRead(18);

		if (iFinCourseMoteur == 0 || iFinCourseEncodeur == 0)
		{
			iSortieBute = 1;
			puts("DANGER Niveau 3 sur une butée mécanique, on arrete!!");

		}
		// fin mesure fin de course

		// Deb verif fin de boucle demandee
		fin = kbhit();
		if (fin != 0)
		{
			carac = fgetc(stdin);
			if (carac == 'q')
				fin = 1;
			else
				fin = 0;
		}
		// fin verif fin de boucle demandee	
        fflush(stdout);

		itmp++;

	}
   	printf("Demande de sortie de la boucle....avec butee hard = %d\n",iSortieBute);
	printf("Byebye!\nTps moyen par iter = %-10.4e\n",mttot/mnloops);
	puts("!! MODE LINEAIRE => OFF !!");
   	nonblock(NB_DISABLE);
   	
   	//On remet tout a zero et on ferme le GPIO
	//Le moteur et la direction
	gpioWrite(16,0);
	gpioPWM(24,0);

   	//On sauvegarde
   	fichier = fopen("debugLineaire.csv","w+");
   	for (k=0 ; k<(itmp-1);k++)
	{
		fprintf(fichier,"%g,%g,%g,%g,%g,%g,%g,%g\n",debug[0][k],debug[1][k],debug[2][k],debug[3][k],debug[4][k],debug[5][k],debug[6][k],debug[7][k]);
		//printf("iter %d ok!\n",k);
    }
	fclose(fichier);


	//Les encodeurs (fin interruption)
   	Pi_Renc_cancel(enc_moteur);
   	Pi_Renc_cancel(enc_pendule);
   	
   	gpioTerminate();

    
	return 0;

}


/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////


void callback_moteur(int way)
{
   	int idiff;

	ipuls_chariot += way;
	
	// Calcul du sens. Si sens = 1 on va vers l'encodeur, si 0 on va vers le moteur
	idiff = ipuls_chariot - ipuls_chariot_old;

   	if ( idiff > 0 )
		isens_chariot = 1;
   	else
	 	isens_chariot = 0;

	ipuls_chariot_old = ipuls_chariot;
      	      
}

/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
//
void callback_pendule(int way)
{
   	int idiff;

	ipuls_pendule += way;
	
	// Calcul du sens. Si sens = 1 si trigo (antihoraire), si 0 si anti trigo (horaire)
	idiff = ipuls_pendule - ipuls_pendule_old;

   	if ( idiff > 0 )
		isens_pendule = 1;
   	else
	 	isens_pendule = 0;

	ipuls_pendule_old = ipuls_pendule;
      	      
}

/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////

int kbhit()
{
    struct timeval tv;
    fd_set fds;
    tv.tv_sec = 0;
    tv.tv_usec = 0;
    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds); //STDIN_FILENO is 0
    select(STDIN_FILENO+1, &fds, NULL, NULL, &tv);
    return FD_ISSET(STDIN_FILENO, &fds);
}

/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////

void nonblock(int state)
{
    struct termios ttystate;

    //get the terminal state
    tcgetattr(STDIN_FILENO, &ttystate);

    if (state==NB_ENABLE)
    {
        //turn off canonical mode
        ttystate.c_lflag &= ~ICANON;
        //minimum of number input read.
        ttystate.c_cc[VMIN] = 1;
    }
    else if (state==NB_DISABLE)
    {
        //turn on canonical mode
        ttystate.c_lflag |= ICANON;
    }
    //set the terminal attributes.
    tcsetattr(STDIN_FILENO, TCSANOW, &ttystate);

}
