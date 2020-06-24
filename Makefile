all:
	mpicc main.c -o main-parallel -lm
	gcc serial.c -o main-serial -lm
clean:
	rm main-parallel main-serial