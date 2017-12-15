#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <limits.h>
#include "time.h"

const int trial_num = 10240;

double unidev();

int main(int argc, char** argv)
{
  int seed;
  int count = 4;
  int i;
  int Door[4] = {0};
  int Door_rem[2] = {0}; // The two remaining doors for choice b
  int door_win;
  int door_choice;
  int door_choice_b;
  int elim_door;
  double uni_rnd;

  int k;

  int win_a = 0;
  int win_b = 0;

  /* initialize random number generator */
  seed = (int)time(NULL);
  srandom(seed);

  // Time variable
  clock_t t_global_start, t_global_end;
  double t_global;

  for (k = 0; k < trial_num; ++k)
  {

    // printf("Trial number %d \n", k);


    double r = unidev();
    // printf("r = %f \n", r);

    if (r < 0.25) {
      Door[0] = 1;
      door_win = 1;
    }
    else if (r > 0.25 && r < 0.5) {
      Door[1] = 1;
      door_win = 2;
    }
    else if ( r > 0.5 && r < 0.75) {
      Door[2] = 1;
      door_win = 3;
    }
    else  {
      Door[3] = 1;
      door_win = 4;
    }

    // for (i = 0; i < count; ++i)
    // {
    //   printf("Door[%i] = %i \n", i, Door[i]);
    // }

    double m = unidev();
    // printf("m = %f \n", m);

    if (m < 0.25) {
      door_choice = 1;
    }
    else if (m > 0.25 && m < 0.5) {
      door_choice = 2;
    }
    else if ( m > 0.5 && m < 0.75) {
      door_choice = 3;
    }
    else  {
      door_choice = 4;
    }

    // printf("Winning door is %i \n", door_win);
    // printf("Chosen door is %i \n", door_choice);

    for (i = 0; i < count; ++i)
    {
      if (Door[i] == 0 && door_choice != (i + 1))  {
        elim_door = i + 1;
        break;
      }
    }

    // printf("Eliminated Door number %i \n", elim_door);

    int j = 0;
    for (i = 0; i < count; ++i)
    {
      if (elim_door != (i + 1) && door_choice != (i + 1))
      {
        Door_rem[j] = (i + 1);
        j++;
      }
    }

    // Choice a
    // printf("Choice A was = %d \n", door_choice);
    // (door_win == door_choice) ? printf("Congrats!\n") : printf("Loser!\n");
    if (door_win == door_choice) win_a++;

    // Choice b
    double b_choice = unidev();
    if (b_choice < 0.5) {
      door_choice_b = Door_rem[0];
    }
    else  {
      door_choice_b = Door_rem[1];
    }

    // printf("Choice B was = %d \n\n", door_choice_b);
    // (door_win == door_choice_b) ? printf("Congrats!\n") : printf("Loser!\n");
    if (door_win == door_choice_b) win_b++;

  }

  t_global_end = clock();
  t_global = (double)(t_global_end - t_global_start) / CLOCKS_PER_SEC;

  printf("It took %f sec to complete \n", t_global);


  float pa = (float)win_a / (float)trial_num;
  float pb = (float)win_b / (float)trial_num;

  printf("Number of trials was = %d \n", trial_num);

  printf("Choice A won %d times \n", win_a);
  printf("P_a = %f \n", pa);

  printf("Choice A won %d times \n", win_b);
  printf("P_b = %f \n", pb);

}

double unidev()
{
  /* Generate a uniform number from [0,1]*/
  return (double)random() / (double)RAND_MAX;
}

