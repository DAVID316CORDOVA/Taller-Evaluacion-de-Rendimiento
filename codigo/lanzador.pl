#!/usr/bin/perl
#**************************************************************
#         		Pontificia Universidad Javeriana
#     Autor: J. Corredor (modificado por Grupo 02)
#     Materia: Computación de Alto Rendimiento
#     Tema: Taller de Evaluación de Rendimiento MPI
#     Fichero: script automatización ejecución por lotes
#
#     Batería alineada con la documentada en el informe:
#       - Procesos MPI (np):  2, 3, 4, 5  (master + 1..4 workers)
#       - Tamaños de matriz: 500, 1000, 2000, 3000
#       - Hilos OpenMP:      1, 2, 4
#       - Repeticiones:      30 por configuración
#
#     Restricción: N debe ser divisible por (np-1). Las combinaciones
#     N=500 / np=4 (workers=3) y N=1000 / np=4 se omiten porque no
#     cumplen divisibilidad; verificarDiv() abortaría la ejecución.
#****************************************************************/

use strict;
use warnings;

my $Path = `pwd`;
chomp($Path);

my @Nombre_Ejecutable = ("mxmOmpMPIfxc", "mxmOmpMPIfxt");
my @numProcesos       = (2, 3, 4, 5);
my @Size_Matriz       = (500, 1000, 2000, 3000);
my @Num_Hilos         = (1, 2, 4);
my $Repeticiones      = 30;
my $Resultados        = "Soluciones";
my $filehost          = "filehost";

# Crear carpeta de resultados si no existe
mkdir($Resultados) unless -d $Resultados;

foreach my $nombre (@Nombre_Ejecutable) {
    foreach my $np (@numProcesos) {
        my $workers = $np - 1;
        foreach my $size (@Size_Matriz) {
            # Saltar combinaciones donde N no es divisible por workers
            if ($size % $workers != 0) {
                printf("[SKIP] %s N=%d np=%d (workers=%d no divide a N)\n",
                       $nombre, $size, $np, $workers);
                next;
            }
            foreach my $hilo (@Num_Hilos) {
                my $file = "$Path/$Resultados/$nombre-${size}-np_${np}-Hilos-${hilo}.dat";
                printf("[RUN ] %s\n", $file);
                # Si ya existen 30 líneas, saltar (permite reanudar batería interrumpida)
                if (-e $file) {
                    my $lineas = `wc -l < "$file"`;
                    chomp($lineas);
                    if ($lineas >= $Repeticiones) {
                        printf("       ya tiene $lineas reps, se omite\n");
                        next;
                    }
                }
                for (my $i = 0; $i < $Repeticiones; $i++) {
                    system("mpirun -hostfile $filehost -np $np ./$nombre $size $hilo >> $file");
                }
            }
        }
    }
}

print "\n[FIN] Batería completa. Resultados en $Path/$Resultados/\n";
