#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pigpio.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <sys/types.h>
#include <termios.h>

#include "rotary_encoder.h"

#define PI 3.1415
#define SLEEP 50000	//50ms
#define DODO 1000000 //1s
#define TAILLE_MAX 25000
#define NB_ENABLE  1
#define NB_DISABLE 0

/*
 *
 * Ce programe permet d'identifier les parametres du moteur pour trouver
 * la relation entre la pwm et la force.
 * 
 * Pour le cablage :
 * encodeur 1 (position du chariot) : cable vert => GPIO21 et cable blanc => GPIO20)
 * moteur (signal PWM) : fil jaune => GPIO24
 * moteur (signal DIR) : fil vert => GPIO16
 * 
 * 
 * Pour compiler :
 * 	gcc -o identifMoteur identifMoteur.c rotary_encoder.c -lpigpio -lrt
 *
 * Pour executer :
 * 	sudo ./identifMoteur
 */


//Pour pouvoir soir sur appui touche
int kbhit(void);
void nonblock(int);

void callback_moteur(int);


// Declaration et init des variables
int ipuls_chariot = 0;
int isens_chariot = 0;

// Init ici a zéro de la position, cad cote moteur
// 0m cote moteur, 89cm cote encodeur)
float fpos_chariot = 0.;
float fPoseTempChariot = 0.;


float fcontrol;
int icontrol;
int idirection;


int main(int argc, char *argv[])
{
        //Val PWM pour identif
        int nbValPWM;
        int iPWM[7]={50,75,100,150,200,225,250};
        float poseSave[4][TAILLE_MAX];
        char str1[5]="pwm-";
        char str2[5]="rev-";
        char str3[4];
        char str4[5]=".csv";
        char name1[14]="\0";
        char name2[19]="\0";
        FILE *fichier;
        
        //Pour la sortie
	int fin = 0 ;
	char carac;
	nonblock(NB_ENABLE);

        //variables diverses
	int i,j,k;
	int ikey = 0;
    long mnloops = 0;

	//pour le temps
	double Te = (double)SLEEP*1e-6;

   	//Declaration de l'encodeur chariot
   	Pi_Renc_t *enc_moteur;
	
   	//Init du GPIO, sortie si erreur
   	if (gpioInitialise() < 0) 
   	{
		printf("Pb init gpioInit\n");
		return 1;
   	}
   	printf("GPIO OK\n");

	
	//direction moteur
	gpioSetMode(16, PI_OUTPUT); 


   	enc_moteur = Pi_Renc(20, 21, callback_moteur);
   	
        //Calcul du nb de valeur de PWM dans le tableau
   	nbValPWM = (int)sizeof(iPWM)/(int)sizeof(iPWM[0]);
	printf("Nb elements = %d\n",nbValPWM);
    
   	
   	//HYPOTHESE : position de départ à 10cm de la butée gauche
	for (i=0 ; i<=nbValPWM-1 && !fin; i++)
	{
                // Divers pour le temps de calcul (DEBUT)
		time_t mnow;
		struct timeval mtv;
		struct timezone mtz;
		time(&mnow);
		gettimeofday(&mtv, &mtz);
		double startTime;
		double mttot = 0.0;

		
		sprintf(str3, "%d", iPWM[i]);
		strcat(name1,str1);
		strcat(name1,str3);
		strcat(name1,str4);    	
		fichier = fopen(name1,"w+"); 
		//Iter, PWM, Temps, Position
		
		//Pause 1s
		usleep(DODO);

		//Init intiale
		fPoseTempChariot = 0.;
		ipuls_chariot = 0;
		j=0;
		
		while (fpos_chariot <= 0.5 && !fin)
		{
			time(&mnow);
			gettimeofday(&mtv, &mtz);
			startTime = (double)mnow + mtv.tv_usec/1e6;
					
			//On envoie la commande sur le moteur dans la direction 0
			idirection = 1;
			icontrol = iPWM[i];	
						
			//On definit le sens
			gpioWrite(16,idirection);
			//On envoie au moteur
			gpioPWM(24,icontrol);

			
			//temps de sommeil en microseconde
			usleep(SLEEP);
			
			time(&mnow);
			gettimeofday(&mtv, &mtz);
			Te = (((double)mnow + mtv.tv_usec/1e6) - startTime);
			mttot += Te;
			
			//Mesure de la position 
			//n_step = 600, diametre de l'axe = 6.6mm
			fpos_chariot = ((ipuls_chariot * 2 * PI) / 600 )* 0.006371;
			fPoseTempChariot = ((ipuls_chariot * 2 * PI) / 600 )* 0.006371;
			
			if (j==0)
                        {
                                poseSave[0][j] = 0;
                                poseSave[1][j] = iPWM[i];
                                poseSave[2][j] = mttot-Te;
                                poseSave[3][j] = fPoseTempChariot;
                        }
                        else
                        {
                                poseSave[0][j] = j;	
                                poseSave[1][j] = iPWM[i];	
                                poseSave[2][j] = mttot-Te;	
                                poseSave[3][j] = fPoseTempChariot;	
                        }
                        j++;
        			
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
                
		}
		
		//On coupe tout 
		gpioPWM(24,0);

		printf("Sauvergarde dans %s puis Dodo de 1s!\n",name1);
                for (k=0;k<j;k++)
                {
                        fprintf(fichier,"%g,%g,%g,%g\n",poseSave[0][k],poseSave[1][k],poseSave[2][k],poseSave[3][k]);
                }

                fclose(fichier);
		name1[0]='\0';

		
		//Pause 2s
		usleep(DODO);
		
		/////////////////////
        strcat(name2,str1);
        strcat(name2,str2);
		strcat(name2,str3);
		strcat(name2,str4);    	


		fichier = fopen(name2,"w+");
		//Iter, -PWM, Temps, Position

		fPoseTempChariot = 0.;
		ipuls_chariot = 0;
		j=0;
		mttot = 0.0;
		
		// Et on repart de l'autre cote
		while (fpos_chariot >=-0.5 && !fin)
		{
			time(&mnow);
			gettimeofday(&mtv, &mtz);
			startTime = (double)mnow + mtv.tv_usec/1e6;
			
			//On envoie la commande sur le moteur dans la direction 0
			idirection = 0;
			icontrol = iPWM[i];	
						
			//On definit le sens
			gpioWrite(16,idirection);
		
			//On envoie au moteur
			gpioPWM(24,icontrol);
			
			//temps de sommeil en microseconde
			usleep(SLEEP);
			
			time(&mnow);
			gettimeofday(&mtv, &mtz);
			Te = (((double)mnow + mtv.tv_usec/1e6) - startTime);
			mttot += Te;
			
			//Mesure de la position 
			//n_step = 600, diametre de l'axe = 6.6mm
			fpos_chariot = ((ipuls_chariot * 2 * PI) / 600 )* 0.006371;
			fPoseTempChariot = ((ipuls_chariot * 2 * PI) / 600 )* 0.006371;
	
			if (j==0)
			{
				poseSave[0][j] = 0;
				poseSave[1][j] = -iPWM[i];
				poseSave[2][j] = mttot-Te;
				poseSave[3][j] = fPoseTempChariot;
			}
			else
			{
				poseSave[0][j] = j;	
				poseSave[1][j] = -iPWM[i];	
				poseSave[2][j] = mttot-Te;	
				poseSave[3][j] = fPoseTempChariot;	
			}
			j++;
    		
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
		}
		
		//On coupe tout 
		gpioPWM(24,0);
		
		printf("Sauvergarde dans %s puis Dodo de 1s!\n",name2);
                for (k=0;k<j;k++)
                {
                        fprintf(fichier,"%g,%g,%g,%g\n",poseSave[0][k],poseSave[1][k],poseSave[2][k],poseSave[3][k]);
                }
                name2[0]='\0';

                fclose(fichier);

		//Pause 1s
		usleep(DODO);
		
	}

	printf("Byebye!\n");
        nonblock(NB_DISABLE);

   	//On remet tout a zero et on ferme le GPIO
	//Le moteur et la direction
	gpioWrite(16,0);
	gpioPWM(24,0);
        
        
	//Les encodeurs (fin interruption)
   	Pi_Renc_cancel(enc_moteur);
   	gpioTerminate();

	return 0;

}


/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
void callback_moteur(int way)
{
   	int idiff;

	ipuls_chariot += way;
      	      
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
