
/* ============================================================================
 * ============================================================================
 * CPSC 533b Algorithmic Animation / Final Project
 *
 * file: flowanim.cpp
 *
 * This program uses the fluid dynamics solver presented
 * in Jos Stam, "A Simple Fluid Solver Based on the FFT",
 * (http://www.dgp.utoronto.ca/people/stam/reality/Research/pub.html)
 *
 * To compile this program, you need "The Fastest Forurier
 * Transform in the West", i.e., the fftw and rfftw libraries
 * and header files. They can be downloaded from
 * http://www.fftw.org. The site also describes how to compile on the
 * win32 operating system using precompiled fftw lib's and dll's
 *
 * The initial codebase for this program was written by Gustav Taxen
 * and can be found on the internet at:
 * http://www.nada.kth.se/~gustavt/fluids/
 *
 * Author: Andrew Nealen, Dept. of Computer Science, UBC Vancouver
 * Date: 04/10/02
 * ============================================================================
 * ============================================================================
 */

#ifdef WIN32
#include <windows.h>
#else
 /* no screenshots on win32 boxes */
#include "png.h"
#endif

#include <math.h>
#include <fftw3.h>
#include <GL/glut.h>

#ifndef M_PI
#define M_PI 3.1415926535898
#endif

/* list of prototypes */
void updateSimulation(void);
void init_FFT(int);
void stable_solve(int,
	float*,
	float*,
	float*,
	float*,
	float,
	float);
void diffuse_matter(int,
	float*,
	float*,
	float*,
	float*,
	float);
void drawVectorField(float*,
	float*,
	int, int, int, int, int);
void displayMainWin(void);
void displayVelBefore(void);
void displayVelAfter(void);
void displayGradientField(void);
void displayVelBeforeFFT(void);
void displayVelAfterFFT(void);
void displayGradientFieldFFT(void);
void displayScalarVort(void);
void displayGradVort(void);
void displayForceVort(void);
void reshapeMainWin(int, int);
void reshapeFieldWin(int, int);
void reshapeVortWin(int, int);
void setVelocitiesToZero(void);
void setDensitiesToZero(void);
void redrawAllWindows(void);
void setForces(void);
void displayFieldWin(void);
void displayVortWin(void);
void keyboard(unsigned char, int, int);
void init(void);
void drag(int, int);
void menuFunc(int);
void initArrays(int);
void createTestCase(int);
void saveState(void);
void restoreState(void);

#ifndef WIN32 /* screenshots only on linux */
// void screenshot(const char*);
// void updateSimulationAndScreenshot(void);
// void startMovie(void);
// void stopMovie(void);
#endif

/* GLOBALS and DEFINES */

/*
 * Size of the simulation grid, i.e. the number of grid
 * points are (DIM x DIM). 64 is default value, which can
 * be modified on the command line
 *
 */
int DIM = 128;
int INITWINSIZE = 1000;

/* some drawing colors */
int RED = 1;
int GREEN = 2;
int BLUE = 4;

/*
 * Simulation time step (can be changed with keypresses),
 * viscosity, dissipation rate of the smoke, injected matter
 * at mouse position, epsilon for gradient computation and
 * scale factor for the confinement term
 *
 */
double dt = .8;
double g_visc = 0.001;
double dissipation = 1; // 0.996;
double matterflow = 40.0;
double g_eps = 0.001;
double g_vortForceScale = 0.08;

int    mainWinWidth, mainWinHeight;
int    fieldWinWidth, fieldWinHeight;
int    vortWinWidth, vortWinHeight;
int    drawVelocities = 0;
int    drawVorticity = 0;
int    paused = 0;
int    fieldVisible = 1;
int    vortVisible = 1;
int    vorticityConfinement = 0;
int    framenum = 0;
int    wireframe = 0;
int    lmx, lmy;
int    mainWin, fieldWin, velBeforeWin, velAfterWin, gradientWin;
int    vortWin, vortScalarWin, vortGradWin, vortForceWin;
int    velBeforeWinFFT, velAfterWinFFT, gradientWinFFT;

double t = 0;

int dim2;

/* arrays for the solver computation */
float *u, *v, *u0, *v0;
/* arrays to hold float data for rendering */
float *ub, *vb, *ua, *va, *grad_u, *grad_v;
/* arrays to hold fftwf_complex data for rendering */
float *ubt, *vbt, *uat, *vat, *grad_ut, *grad_vt;
/* arrays for the smoke density and user forces */
float *rho, *rho0, *u_u0, *u_v0;
/* arrays for saved smoke density, user forces and velocities */
float *rho_s, *uf_s, *vf_s, *u_s, *v_s;
/* arrays for the vorticity computation */
float *ws, *w, *w_grad_u, *w_grad_v;
/* arrays for the vorticity confinement forces */
float *vort_force_u, *vort_force_v;

/*
 * See Jos Stam, "A Simple Fluid Solver Based on the FFT",
 * http://www.dgp.utoronto.ca/people/stam/reality/Research/pub.html
 * for more details on the stable_solve function
 *
 * i added A LOT of comments to further document the method and
 * make it a bit clearer
 */

static fftwf_plan plan_rcu, plan_rcv, plan_cru, plan_crv;


#define floor(x) ((x)>=0.0?((int)(x)):(-((int)(1-(x)))))

fftwf_complex* cu;
fftwf_complex* cv;
float* r;

void init_FFT(int n) {

	u0 = (float*)fftwf_alloc_real(n * n * sizeof(float));
	v0 = (float*)fftwf_alloc_real(n * n * sizeof(float));

	cu = fftwf_alloc_complex(n * n * sizeof(fftwf_complex));
	cv = fftwf_alloc_complex(n * n * sizeof(fftwf_complex));

	plan_rcu = fftwf_plan_dft_r2c_2d(n, n, u0, cu, FFTW_PATIENT);
	plan_rcv = fftwf_plan_dft_r2c_2d(n, n, v0, cv, FFTW_PATIENT);

	plan_cru = fftwf_plan_dft_c2r_2d(n, n, cu, u0, FFTW_PATIENT);
	plan_crv = fftwf_plan_dft_c2r_2d(n, n, cv, v0, FFTW_PATIENT);
}

void stable_solve(int n,
	float * u,
	float * v,
	float * u0,
	float * v0,
	float visc,
	float dt) {

	float x, y, f, r_sq, U[2], V[2], s, t;
	float vorticity, mag, gu, gv;
	int i, j, i0, j0, i1, j1, i_1, j_1;

	/* apply user defined forces (f) stored in u0 and v0
	 * only add the confinement forces (vort_force_u and
	 * vort_force_v) if they are activated by the user
	 */
	if (vorticityConfinement) {
		for (i = 0; i < n*n; i++) {
			u[i] += dt * u0[i] + dt * vort_force_u[i];
			u0[i] = u[i];

			v[i] += dt * v0[i] + dt * vort_force_v[i];
			v0[i] = v[i];
		}
	}
	else {
		for (i = 0; i < n*n; i++) {
			u[i] += dt * u0[i];
			u0[i] = u[i];

			v[i] += dt * v0[i];
			v0[i] = v[i];
		}
	}

	/* advection step (-(u.G).u) */
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			/* compute exact position of particle one timestep ago */
			x = i - dt * u0[i + n * j] * n; y = j - dt * v0[i + n * j] * n;
			/* discretize to grid and compute interpolation factors s and t */
			i0 = floor(x);
			s = x - i0;
			j0 = floor(y);
			t = y - j0;
			/* make sure that advection wraps in u direction */
			i0 = (n + (i0%n)) % n;
			i1 = (i0 + 1) % n;
			/* make sure that advection wraps in v direction */
			j0 = (n + (j0%n)) % n;
			j1 = (j0 + 1) % n;
			/* set new velocity to linear interpolation of previous
		   particle position using interpolation factors s and t
		   and the four grid positions i0, i1, j0 and j1 */
			u[i + n * j] = (1 - s)*((1 - t)*u0[i0 + n * j0] + t * u0[i0 + n * j1]) +
				s * ((1 - t)*u0[i1 + n * j0] + t * u0[i1 + n * j1]);
			v[i + n * j] = (1 - s)*((1 - t)*v0[i0 + n * j0] + t * v0[i0 + n * j1]) +
				s * ((1 - t)*v0[i1 + n * j0] + t * v0[i1 + n * j1]);
		}
	}

	/* swap grids (copy u and v into u0 and v0) */
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			u0[i + n*j] = u[i + n * j];
			v0[i + n*j] = v[i + n * j];
		}
	}

	/* copy velocity field before projection and diffusion so it
	   can be rendered later on (using ub, vb) */
	if (fieldVisible) {
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				ub[i + n * j] = u0[i + n*j];
				vb[i + n * j] = v0[i + n*j];
			}
		}
	}

	/* fourier transform forward */
	fftwf_execute(plan_rcu);
	fftwf_execute(plan_rcv);

	/* copy FFT of velocity field before projection and diffusion so it
	   can be rendered later on (using ubt, vbt) */
	if (fieldVisible) {
		for (i = 0; i <= n; i += 1) {
			for (j = 0; j < n; j++) {
				/* real part on the right */
				ubt[dim2 + i + n * ((j + dim2) % n)] = cu[i + n*j][0];
				/* imaginary part on the left */
				ubt[i + n * ((j + dim2) % n)] = cu[i + n*j][1];

				/* real part on the right */
				vbt[dim2 + i + n * ((j + dim2) % n)] = cv[i + n * j][0];
				/* imaginary part on the left */
				vbt[i + n * ((j + dim2) % n)] = cv[i + n * j][1];

			}
		}
	}


	/* ----------------------------------------------------
	 * diffusion and projection steps in the fourier domain
	 * ----------------------------------------------------
	 *
	 * x(i) runs in the interval [0,n/2] and only touches the real parts
	 * of the complex fourier transform (i+=2)
	 *
	 * y(j) runs in the interval [-n/2,n/2], but runs from 0 to n/2 and
	 * from -n/2 back up to zero (skewed by n/2)
	 *
	 * output of FFTW looks like this (all types are float's, but
	 * they differ in 'real' (r) and 'imaginary' (i) part of the transform
	 *
	 *       x -> [0,n/2]
	 *
	 *         0   1   2     . . .      n/2
	 *
	 * y  0    r i r i r i r i r i . . . r i
	 * |  1    r i r i r i r i r i . . . r i
	 * v       . . . . . . . . . . . . . . .
	 *    n/2  r i r i r i r i r i . . . r i
	 *  - n/2  r i r i r i r i r i . . . r i
	 *         . . . . . . . . . . . . . . .
	 *  - 1    r i r i r i r i r i . . . r i
	 *    0    r i r i r i r i r i . . . r i
	 *
	 */
	for (i = 0; i <= n; i += 1) {
		/* x runs in the interval [0,n/2] */
		x = 0.5*i;
		for (j = 0; j < n; j++) {
			/* x runs in the interval [-n/2,n/2] but in the sequence (from
		   top to bottom) 0, n/2, -n/2, 0 */
			y = j <= n / 2 ? j : j - n;
			/* r_sq is sq distance from the origin */
			r_sq = x * x + y * y;
			if (r_sq == 0.0) continue;
			/* low-pass filter (viscosity/dampening) */
			f = exp(-r_sq * dt*visc);

			/* U[0] is real, U[1] is imaginary part of u transform
			 * V[0] is real, V[1] is imaginary part of v transform
			 */
			U[0] = cu[i + n * j][0]; 			U[1] = cu[i + n * j][1];
			V[0] = cv[i + n * j][0]; 			V[1] = cv[i + n * j][1];

			/* project both the real and imaginary parts of the complex transform
			 * onto lines perpendicular to the corresponding wavenumber and
			 * multiply with the low-pass filter f. this line would
			 * be a (hyper)plane in higher dimensions.
			 */
			cu[i + n*j][0] = f * ((1 - x * x / r_sq)*U[0] + -x * y / r_sq * V[0]);
			cu[i + n*j][1] = f * ((1 - x * x / r_sq)*U[1] + -x * y / r_sq * V[1]);
			cv[i + n*j][0] = f * (-y * x / r_sq * U[0] + (1 - y * y / r_sq)*V[0]);
			cv[i + n*j][1] = f * (-y * x / r_sq * U[1] + (1 - y * y / r_sq)*V[1]);
		}
	}

	if (fieldVisible) {
		/* copy FFT of velocity field after projection and diffusion so it
		   can be rendered later on (using uat, vat) */
		for (i = 0; i <= n; i += 1) {
			for (j = 0; j < n; j++) {
				/* real part right */
				uat[dim2 + i + n * ((j + dim2) % n)] = cu[i + n*j][0];
				vat[dim2 + i + n * ((j + dim2) % n)] = cv[i + n*j][0];
				/* imaginary part left*/
				uat[i + n * ((j + dim2) % n)] = cu[i + n*j][1];
				vat[i + n * ((j + dim2) % n)] = cv[i + n*j][1];
			}
		}

		/* compute FFT of gradient field and store in grad_ut and grad_vt */
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				grad_ut[i + n * j] = -ubt[i + n * j] + uat[i + n * j];
				grad_vt[i + n * j] = -vbt[i + n * j] + vat[i + n * j];
			}
		}
	}

	/* inverse fourier transform */
	fftwf_execute(plan_cru);
	fftwf_execute(plan_crv);

	if (fieldVisible) {
		/* copy velocity field after projection and diffusion so it
		   can be rendered later on (using ua, va) */
		f = 1.0 / (n*n);
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				ua[i + n * j] = f * u0[i + n*j];
				va[i + n * j] = f * v0[i + n*j];
			}
		}

		/* compute gradient field and store in grad_u and grad_v */
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				grad_u[i + n * j] = ub[i + n * j] - ua[i + n * j];
				grad_v[i + n * j] = vb[i + n * j] - va[i + n * j];
			}
		}
	}

	/* normalization step:
	 * The RFFTWND transforms are unnormalized, so a forward followed
	 * by a backward transform will result in the original data scaled
	 * by the number of real data elements--that is, the product of the
	 * (logical) dimensions of the real data.
	 */
	f = 1.0 / (n*n);
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			u[i + n * j] = f * u0[i + n*j];
			v[i + n * j] = f * v0[i + n*j];
		}
	}

	/* --------------------------
	 * vorticity confinement step
	 * --------------------------
	 *
	 * compute vorticity field (ws) and store as a scalar field of the
	 * vorticity vector magnitude
	 */
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			/* make sure that finite differencing of vorticity wraps:
			 * i1 = index of cell i+1, i_1 = index of cell i-1 (same for j)
			 */
			i_1 = i - 1; i_1 = (n + (i_1%n)) % n;
			i1 = i + 1; i1 = i1 % n;
			j_1 = j - 1; j_1 = (n + (j_1%n)) % n;
			j1 = j + 1; j1 = j1 % n;

			/* compute vorticity as finite difference dv/dx - du/dy
			 * which corresponds to the cross product of the gradient
			 * operator and the velocity vector (see project report)
			 */

			 /* simple forward differencing
			vorticity = (v[i1+n*j]-v[i+n*j]) - (u[i+n*j1]-u[i+n*j]);
			 */

			 /* alternatively (and better), use central differencing */
			vorticity = ((v[i1 + n * j] - v[i_1 + n * j]) - (u[i + n * j1] - u[i + n * j_1])) / 2;

			/* store vorticity magnitude (for gradient computation) */
			ws[i + n * j] = (vorticity < 0.0f) ? -1.0f*vorticity : vorticity;

			/* store true vorticity with sign (vector perpendicular
		   to the u/v plane) */
			w[i + n * j] = vorticity;
		}
	}

	/* compute gradient field of vorticity magnitude and normalize */
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			i_1 = i - 1; i_1 = (n + (i_1%n)) % n;
			i1 = i + 1; i1 = i1 % n;
			j_1 = j - 1; j_1 = (n + (j_1%n)) % n;
			j1 = j + 1; j1 = j1 % n;

			/* compute the central difference (gu, gu are gradient vec's) */
			gu = (ws[i1 + n * j] - ws[i_1 + n * j]) / 2;
			gv = (ws[i + n * j1] - ws[i + n * j_1]) / 2;

			/* normalize */
			mag = sqrt(gu*gu + gv * gv);

			if (mag < g_eps) {
				w_grad_u[i + n * j] = w_grad_v[i + n * j] = 0.0;
			}
			else {
				w_grad_u[i + n * j] = gu / mag;
				w_grad_v[i + n * j] = gv / mag;
			}
		}
	}

	/* compute forces to be applied if vorticity confinement is enabled */
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			vort_force_u[i + n * j] = w[i + n * j] * w_grad_v[i + n * j] * g_vortForceScale;
			vort_force_v[i + n * j] = -w[i + n * j] * w_grad_u[i + n * j] * g_vortForceScale;
		}
	}
}

/*
 * This function diffuses matter that has been placed
 * in the velocity field. It's almost identical to the
 * velocity advection step in the function above. The
 * input matter densities are in rho0 and the result
 * is written into rho.
 *
 */
void diffuse_matter(int n,
	float *u,
	float *v,
	float *rho,
	float *rho0,
	float dt) {

	float x, y, s, t;
	int i, j, i0, j0, i1, j1;

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			/* compute exact position of particle one timestep ago */
			x = i - dt * u[i + n * j] * n; y = j - dt * v[i + n * j] * n;
			/* discretize to grid and compute interpolation factors s and t */
			i0 = floor(x);
			s = x - i0;
			j0 = floor(y);
			t = y - j0;
			/* make sure that advection wraps in u direction */
			i0 = (n + (i0%n)) % n;
			i1 = (i0 + 1) % n;
			/* make sure that advection wraps in v direction */
			j0 = (n + (j0%n)) % n;
			j1 = (j0 + 1) % n;
			/* set new density to the linear interpolation of previous
		   density (delta t ago) using interpolation factors s and t
		   and the four grid positions i0, i1, j0 and j1 */
			rho[i + n * j] = (1 - s)*((1 - t)*rho0[i0 + n * j0] + t * rho0[i0 + n * j1]) +
				s * ((1 - t)*rho0[i1 + n * j0] + t * rho0[i1 + n * j1]);
		}
	}
}

/*
 * Draw a vector field u,v in the windowsize specified using
 * GL_LINES
 *
 */
void drawVectorField(float *u,
	float *v,
	int winWidth,
	int winHeight,
	int scale,
	int color_s,
	int color_e) {

	int        i, j, idx;
	/* Grid element width */
	float  wn = (float)winWidth / (float)(DIM + 1);
	/* Grid element height */
	float  hn = (float)winHeight / (float)(DIM + 1);

	glBegin(GL_LINES);
	for (i = 0; i < DIM; i++) {
		for (j = 0; j < DIM; j++) {
			idx = (j * DIM) + i;
			glColor3f((RED & color_s) ? 1.0f : 0.0f,
				(GREEN & color_s) ? 1.0f : 0.0f,
				(BLUE & color_s) ? 1.0f : 0.0f);
			glVertex2f(wn + (float)i * wn, hn + (float)j * hn);
			glColor3f((RED & color_e) ? 1.0f : 0.0f,
				(GREEN & color_e) ? 1.0f : 0.0f,
				(BLUE & color_e) ? 1.0f : 0.0f);
			glVertex2f((wn + (float)i * wn) + scale * u[idx],
				(hn + (float)j * hn) + scale * v[idx]);
		}
	}
	glEnd();
}

/*
 * Draw scalar field s using GL_TRIANGLE_STRIP's for each column
 *
 */
void drawScalarField(float* s,
	int winWidth,
	int winHeight,
	int color,
	int scale,
	float alpha) {
	int        i, j, idx;
	/* Grid element width */
	float  wn = (float)winWidth / (float)(DIM + 1);
	/* Grid element height */
	float  hn = (float)winHeight / (float)(DIM + 1);
	double     px, py;

	/* ifdef here because of some heinous bug. wireframe works
	just fine under linux but garbles up on win32 os's */
#ifdef WIN32
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
#else
	if (wireframe) {
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	}
	else {
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
#endif

	for (j = 0; j < DIM - 1; j++) {
		glBegin(GL_TRIANGLE_STRIP);

		i = 0;
		px = wn + (float)i * wn;
		py = hn + (float)j * hn;
		idx = (j * DIM) + i;
		glColor4f((RED & color) ? scale * s[idx] : 0.0f,
			(GREEN & color) ? scale * s[idx] : 0.0f,
			(BLUE & color) ? scale * s[idx] : 0.0f, alpha);
		glVertex2f(px, py);

		for (i = 0; i < DIM - 1; i++) {
			px = wn + (float)i * wn;
			py = hn + (float)(j + 1) * hn;
			idx = ((j + 1) * DIM) + i;
			glColor4f((RED & color) ? scale * s[idx] : 0.0f,
				(GREEN & color) ? scale * s[idx] : 0.0f,
				(BLUE & color) ? scale * s[idx] : 0.0f, alpha);
			glVertex2f(px, py);

			px = wn + (float)(i + 1) * wn;
			py = hn + (float)j * hn;
			idx = (j * DIM) + (i + 1);
			glColor4f((RED & color) ? scale * s[idx] : 0.0f,
				(GREEN & color) ? scale * s[idx] : 0.0f,
				(BLUE & color) ? scale * s[idx] : 0.0f, alpha);
			glVertex2f(px, py);
		}

		px = wn + (float)(DIM - 1) * wn;
		py = hn + (float)(j + 1) * hn;
		idx = ((j + 1) * DIM) + (DIM - 1);
		glColor4f((RED & color) ? scale * s[idx] : 0.0f,
			(GREEN & color) ? scale * s[idx] : 0.0f,
			(BLUE & color) ? scale * s[idx] : 0.0f, alpha);
		glVertex2f(px, py);

		glEnd();
	}
}

void displayMainWin(void) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	/* either draw vorticity or density (smoke) field */
	if (!drawVorticity) {
		drawScalarField(rho,
			mainWinWidth,
			mainWinHeight,
			RED | GREEN | BLUE,
			1, 1.0);
	}
	else {
		drawScalarField(ws,
			mainWinWidth,
			mainWinHeight,
			RED | GREEN,
			500, 1.0);
	}

	if (drawVelocities) {
		drawVectorField(u, v,
			mainWinWidth, mainWinHeight,
			1000, RED, RED);
	}

	glFlush();
	glutSwapBuffers();
}

void displayVelBefore(void) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	drawVectorField(ub, vb, fieldWinWidth / 3,
		fieldWinHeight / 2,
		1000, RED, RED);

	glFlush();
	glutSwapBuffers();
}

void displayVelAfter(void) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	drawVectorField(ua, va, fieldWinWidth / 3,
		fieldWinHeight / 2,
		1000, RED, RED);

	glFlush();
	glutSwapBuffers();
}

void displayGradientField(void) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	drawVectorField(grad_u, grad_v, fieldWinWidth / 3,
		fieldWinHeight / 2,
		3000, RED, RED);

	glFlush();
	glutSwapBuffers();
}

void displayVelBeforeFFT(void) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	drawVectorField(ubt, vbt, fieldWinWidth / 3,
		fieldWinHeight / 2,
		20, RED, RED);

	glFlush();
	glutSwapBuffers();
}

void displayVelAfterFFT(void) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	drawVectorField(uat, vat, fieldWinWidth / 3,
		fieldWinHeight / 2,
		20, RED, RED);

	glFlush();
	glutSwapBuffers();
}

void displayGradientFieldFFT(void) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	drawVectorField(grad_ut, grad_vt, fieldWinWidth / 3,
		fieldWinHeight / 2,
		60, RED, RED);

	glFlush();
	glutSwapBuffers();
}

void displayScalarVort(void) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	drawScalarField(ws,
		vortWinWidth / 3,
		vortWinHeight,
		GREEN | RED,
		500, 1.0);

	drawVectorField(u, v, vortWinWidth / 3,
		vortWinHeight, 1000,
		RED, RED);

	glFlush();
	glutSwapBuffers();
}

void displayGradVort(void) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	drawVectorField(w_grad_u, w_grad_v, vortWinWidth / 3,
		vortWinHeight,
		8, RED, RED);

	glFlush();
	glutSwapBuffers();
}

void displayForceVort(void) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	drawVectorField(vort_force_u, vort_force_v, vortWinWidth / 3,
		vortWinHeight,
		50000, RED, BLUE);

	glFlush();
	glutSwapBuffers();
}

void reshapeMainWin(int w, int h) {
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, (GLdouble)w, 0.0, (GLdouble)h);
	mainWinWidth = w;
	mainWinHeight = h;
}

void reshapeFieldWin(int w, int h) {
	glutSetWindow(fieldWin);
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, (GLdouble)w, 0.0, (GLdouble)h);
	fieldWinWidth = w;
	fieldWinHeight = h;

	glutSetWindow(velBeforeWin);
	glViewport(0, 0, w / 3, h / 2);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, (GLdouble)w / 3, 0.0, (GLdouble)h / 2);
	glutPositionWindow(0, 0);
	glutReshapeWindow(fieldWinWidth / 3, fieldWinHeight / 2);

	glutSetWindow(velAfterWin);
	glViewport(0, 0, w / 3, h / 2);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, (GLdouble)w / 3, 0.0, (GLdouble)h / 2);
	glutPositionWindow(fieldWinWidth / 3, 0);
	glutReshapeWindow(fieldWinWidth / 3, fieldWinHeight / 2);

	glutSetWindow(gradientWin);
	glViewport(0, 0, w / 3, h / 2);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, (GLdouble)w / 3, 0.0, (GLdouble)h / 2);
	glutPositionWindow(fieldWinWidth * 2 / 3, 0);
	glutReshapeWindow(fieldWinWidth / 3, fieldWinHeight / 2);

	glutSetWindow(velBeforeWinFFT);
	glViewport(0, 0, w / 3, h / 2);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, (GLdouble)w / 3, 0.0, (GLdouble)h / 2);
	glutPositionWindow(0, fieldWinHeight / 2);
	glutReshapeWindow(fieldWinWidth / 3, fieldWinHeight / 2);

	glutSetWindow(velAfterWinFFT);
	glViewport(0, 0, w / 3, h / 2);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, (GLdouble)w / 3, 0.0, (GLdouble)h / 2);
	glutPositionWindow(fieldWinWidth / 3, fieldWinHeight / 2);
	glutReshapeWindow(fieldWinWidth / 3, fieldWinHeight / 2);

	glutSetWindow(gradientWinFFT);
	glViewport(0, 0, w / 3, h / 2);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, (GLdouble)w / 3, 0.0, (GLdouble)h / 2);
	glutPositionWindow(fieldWinWidth * 2 / 3, fieldWinHeight / 2);
	glutReshapeWindow(fieldWinWidth / 3, fieldWinHeight / 2);
}

void reshapeVortWin(int w, int h) {
	glutSetWindow(vortWin);
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, (GLdouble)w, 0.0, (GLdouble)h);
	vortWinWidth = w;
	vortWinHeight = h;

	glutSetWindow(vortScalarWin);
	glViewport(0, 0, w / 3, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, (GLdouble)w / 3, 0.0, (GLdouble)h);
	glutPositionWindow(0, 0);
	glutReshapeWindow(vortWinWidth / 3, vortWinHeight);

	glutSetWindow(vortGradWin);
	glViewport(0, 0, w / 3, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, (GLdouble)w / 3, 0.0, (GLdouble)h);
	glutPositionWindow(vortWinWidth / 3, 0);
	glutReshapeWindow(vortWinWidth / 3, vortWinHeight);

	glutSetWindow(vortForceWin);
	glViewport(0, 0, w / 3, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, (GLdouble)w / 3, 0.0, (GLdouble)h);
	glutPositionWindow(vortWinWidth * 2 / 3, 0);
	glutReshapeWindow(vortWinWidth / 3, vortWinHeight);
}

void setVelocitiesToZero() {
	int i;
	for (i = 0; i < DIM * DIM; i++) {
		u[i] = v[i] = u0[i] = v0[i] =
			u_u0[i] = u_v0[i] =
			ws[i] = w[i] = w_grad_u[i] = w_grad_v[i] =
			vort_force_u[i] = vort_force_v[i] = 0.0f;
	}
}

void setDensitiesToZero() {
	int i;
	for (i = 0; i < DIM * DIM; i++) {
		rho[i] = rho0[i] = 0.0f;
	}
}

void redrawAllWindows(void) {
	if (fieldVisible) {
		glutSetWindow(fieldWin);
		glutPostRedisplay();

		glutSetWindow(velBeforeWin);
		glutPostRedisplay();

		glutSetWindow(velAfterWin);
		glutPostRedisplay();

		glutSetWindow(gradientWin);
		glutPostRedisplay();

		glutSetWindow(velBeforeWinFFT);
		glutPostRedisplay();

		glutSetWindow(velAfterWinFFT);
		glutPostRedisplay();

		glutSetWindow(gradientWinFFT);
		glutPostRedisplay();
	}

	if (vortVisible) {
		glutSetWindow(vortWin);
		glutPostRedisplay();

		glutSetWindow(vortScalarWin);
		glutPostRedisplay();

		glutSetWindow(vortGradWin);
		glutPostRedisplay();

		glutSetWindow(vortForceWin);
		glutPostRedisplay();
	}

	glutSetWindow(mainWin);
	glutPostRedisplay();
}

/*
 * Copy user-induced forces to the force vectors
 * that is sent to the solver. Also dampen forces and
 * matter density.
 *
 */
void setForces(void) {
	int i;
	for (i = 0; i < DIM * DIM; i++) {
		rho0[i] = dissipation * rho[i];

		u_u0[i] *= 0.85;
		u_v0[i] *= 0.85;

		u0[i] = u_u0[i];
		v0[i] = u_v0[i];
	}
}

/*
 * Update the simulation when we're not drawing.
 *
 */
void updateSimulation(void) {
	setForces();
	stable_solve(DIM, u, v, u0, v0, g_visc, dt);
	diffuse_matter(DIM, u, v, rho, rho0, dt);

	t += dt;

	redrawAllWindows();
}

#ifndef WIN32 /* screenshots only on linux */
// void screenshot(const char* filename) {
// 	/* this allocation does not work under vc++, as
// 	   DIM is not a const */
// 	unsigned char buf[DIM * DIM];
// 	for (int i = 0; i < DIM; i++) {
// 		for (int j = 0; j < DIM; j++) {
// 			float val = rho[i*DIM + j] * 255.0f;
// 			if (val > 255.0f) val = 255.0f;
// 			buf[(DIM - i - 1)*DIM + j] = (unsigned char)val;
// 		}
// 	}
// 	outputGreyscaleImageDataAsPPM(filename, &buf[0], DIM, DIM);
// }

// void updateSimulationAndScreenshot(void) {
// 	setForces();
// 	stable_solve(DIM, u, v, u0, v0, g_visc, dt);
// 	diffuse_matter(DIM, u, v, rho, rho0, dt);

// 	t += dt;

// 	char buf[50];
// 	sprintf(buf, "smoke%d.ppm", framenum);
// 	framenum++;
// 	screenshot(buf);

// 	redrawAllWindows();
// }

// void startMovie(void) {
// 	glutIdleFunc(updateSimulationAndScreenshot);
// }

// void stopMovie(void) {
// 	glutIdleFunc(updateSimulation);
// 	framenum = 0;
// }
#endif

void displayFieldWin(void) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glutSwapBuffers();
}

void displayVortWin(void) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glutSwapBuffers();
}

void createTestCase(int caseNum) {

	int i, j;
	float f;

	setVelocitiesToZero();
	setDensitiesToZero();
	updateSimulation();

	switch (caseNum) {
	case 1:
		/* swirl from upper left to lower right */
		for (i = 0, j = DIM - 1; i < DIM / 3; i++, j--) {
			rho[i + j * DIM] = 2 * matterflow;
			u_u0[i + j * DIM] += 0.1 + ((float)i) * 0.01;
			u_v0[i + j * DIM] -= 0.1 + ((float)i) * 0.01;
		}
		break;
	case 2:
		/* large testcase */
		for (j = DIM / 10; j < DIM - DIM / 10; j++) {
			i = DIM / 10;
			rho[i + j * DIM] = matterflow;
			u_v0[i + j * DIM] += 0.1;
			i = DIM - DIM / 10;
			rho[i + j * DIM] = matterflow;
			u_v0[i + j * DIM] -= 0.1;
		}
		for (i = DIM / 10; i < DIM - DIM / 10; i++) {
			j = DIM / 10;
			rho[i + j * DIM] = matterflow;
			u_u0[i + j * DIM] -= 0.1;
			j = DIM - DIM / 10;
			rho[i + j * DIM] = matterflow;
			u_u0[i + j * DIM] += 0.1;
		}
		float radius = ((float)DIM) / 4.0f;
		for (f = 0.0f; f < 360.0; f += 0.25f) {
			float angle = f / (2 * M_PI);
			float x = sin(angle)*radius;
			float y = cos(angle)*radius;
			rho[(DIM / 2 + floor(x)) + (DIM / 2 + floor(y)) * DIM] =
				2 * matterflow;
			u_u0[(DIM / 2 + floor(x)) + (DIM / 2 + floor(y)) * DIM] -=
				cos(angle)*0.1;
			u_v0[(DIM / 2 + floor(x)) + (DIM / 2 + floor(y)) * DIM] +=
				sin(angle)*0.1;
		}
		break;
	}
}

void keyboard(unsigned char key, int x, int y) {
	int num;
	switch (key) {
	case 'z':
		setVelocitiesToZero();
		updateSimulation();
		break;
	case 'c':
		drawVorticity = 1 - drawVorticity;
		drawVorticity ?
			printf("displaying vorticity\n") :
			printf("displaying smoke density\n");
		break;
	case 'u':
		drawVelocities = 1 - drawVelocities;
		break;
	case 'C':
		vorticityConfinement = 1 - vorticityConfinement;
		if (vorticityConfinement) {
			printf("vorticity confinement enabled\n");
		}
		else {
			printf("vorticity confinement disabled\n");
		}
		break;
	case 'w':
		wireframe = 1 - wireframe;
		break;
	case 't':
		dt -= 0.01;
		printf("dt = %f\n", dt);
		break;
	case 'T':
		dt += 0.01;
		printf("dt = %f\n", dt);
		break;
	case 'v':
		g_visc /= 10.0;
		printf("visc = %f\n", g_visc);
		break;
	case 'V':
		g_visc *= 10.0;
		printf("visc = %f\n", g_visc);
		break;
	case 'd':
		dissipation -= 0.001;
		printf("dissipation rate = %f\n", dissipation);
		break;
	case 'D':
		dissipation += 0.001;
		printf("dissipation rate = %f\n", dissipation);
		break;
	case 's':
		matterflow -= 1.0;
		printf("matter injected = %f\n", matterflow);
		break;
	case 'S':
		matterflow += 1.0;
		printf("matter injected = %f\n", matterflow);
		break;
	case 'O':
		g_vortForceScale += 0.01;
		printf("vorticity force scale factor = %f\n", g_vortForceScale);
		break;
	case 'o':
		g_vortForceScale -= 0.01;
		printf("vorticity force scale factor = %f\n", g_vortForceScale);
		break;
	case 'q':
		exit(0);
		break;
	case 'p':
		glutSetWindow(mainWin);
		glutIdleFunc(paused ? updateSimulation : redrawAllWindows);
		paused = 1 - paused;
		paused ?
			printf("simulation paused\n") : printf("simulation running\n");
		break;
	case 'a':
		setDensitiesToZero();
		updateSimulation();
		break;
	case 'f':
		glutSetWindow(fieldWin);
		fieldVisible ? glutHideWindow() : glutShowWindow();
		fieldVisible = 1 - fieldVisible;
		break;
	case 'F':
		glutSetWindow(vortWin);
		vortVisible ? glutHideWindow() : glutShowWindow();
		vortVisible = 1 - vortVisible;
		break;
	case 'r':
		setVelocitiesToZero();
		setDensitiesToZero();
		updateSimulation();
		break;
	case 'l':
		printf("enter testcase number:\n");
		scanf("%d", &num);
		createTestCase(num);
		break;
	case 'n':
		/* backup the current configuration of forces and densities */
		printf("current state saved\n");
		saveState();
		break;
	case 'N':
		/* restore saved configuration of forces and densities */
		printf("saved state restored\n");
		restoreState();
		break;
#ifndef WIN32
	// 	/* screenshot series functionality. pnglib only works correctly
	// 	   on the linux platform*/
	// case 'i':
	// 	printf("screenshot taken\n");
	// 	screenshot("smoke.pgm");
	// 	break;
	// case 'M':
	// 	printf("movie started\n");
	// 	startMovie();
	// 	break;
	// case 'm':
	// 	printf("movie stopped\n");
	// 	stopMovie();
	// 	break;
#endif
	}
}

void init(void) {
	int i;

	init_FFT(DIM);

	for (i = 0; i < DIM * DIM; i++) {
		u[i] = v[i] = u0[i] = v0[i] = rho[i] = rho0[i] =
			u_u0[i] = u_v0[i] = 0.0f;
	}

	glShadeModel(GL_SMOOTH);
}

/*
 * When the user drags with the mouse, add a force
 * that corresponds to the direction of the mouse
 * cursor movement. Also inject some new matter into
 * the field at the mouse location.
 *
 */
void drag(int mx, int my) {
	int     xi;
	int     yi;
	double  dx, dy, len;
	int     X, Y;

	/* Compute the array index that corresponds to the
	 * cursor location */

	xi = (int)floor((double)(DIM + 1) *
		((double)mx / (double)mainWinWidth));
	yi = (int)floor((double)(DIM + 1) *
		((double)(mainWinHeight - my) / (double)mainWinHeight));

	X = xi;
	Y = yi;

	if (X > (DIM - 1)) {
		X = DIM - 1;
	}
	if (Y > (DIM - 1)) {
		Y = DIM - 1;
	}
	if (X < 0) {
		X = 0;
	}
	if (Y < 0) {
		Y = 0;
	}

	/* Add force at the cursor location */

	my = mainWinHeight - my;
	dx = mx - lmx;
	dy = my - lmy;
	len = sqrt(dx * dx + dy * dy);
	if (len != 0.0) {
		dx *= 0.02 / len;
		dy *= 0.02 / len;
	}
	u_u0[Y * DIM + X] += dx;
	u_v0[Y * DIM + X] += dy;

	/* Increase matter density at the cursor location */

	rho[Y * DIM + X] = matterflow;

	lmx = mx;
	lmy = my;
}

void saveState(void) {
	int i;
	for (i = 0; i < DIM*DIM; i++) {
		uf_s[i] = u_u0[i];
		vf_s[i] = u_v0[i];
		u_s[i] = u[i];
		v_s[i] = v[i];
		rho_s[i] = rho[i];
	}
}

void restoreState(void) {
	int i;
	for (i = 0; i < DIM*DIM; i++) {
		u_u0[i] = uf_s[i];
		u_v0[i] = vf_s[i];
		u[i] = u_s[i];
		v[i] = v_s[i];
		rho[i] = rho_s[i];
	}
}

void menuFunc(int value)
{
	switch (value) {
	case 1:
		glutSetWindow(fieldWin);
		fieldVisible ? glutHideWindow() : glutShowWindow();
		fieldVisible = 1 - fieldVisible;
		break;
	case 2:
		glutSetWindow(mainWin);
		glutIdleFunc(paused ? updateSimulation : redrawAllWindows);
		paused = 1 - paused;
		paused ?
			printf("simulation paused\n") : printf("simulation running\n");
		break;
	case 3:
		exit(0);
		break;
	case 4:
		setVelocitiesToZero();
		setDensitiesToZero();
		updateSimulation();
		break;
	case 5:
		dt += 0.01;
		printf("dt = %f\n", dt);
		break;
	case 6:
		dt -= 0.01;
		printf("dt = %f\n", dt);
		break;
	case 7:
		g_visc *= 10.0;
		printf("visc = %f\n", g_visc);
		break;
	case 8:
		g_visc /= 10.0;
		printf("visc = %f\n", g_visc);
		break;
	case 9:
		setVelocitiesToZero();
		updateSimulation();
		break;
	case 10:
		setDensitiesToZero();
		updateSimulation();
		break;
	case 11:
		matterflow += 1.0;
		printf("matter injected = %f\n", matterflow);
		break;
	case 12:
		matterflow -= 1.0;
		printf("matter injected = %f\n", matterflow);
		break;
	case 13:
		dissipation += 0.001;
		printf("dissipation rate = %f\n", dissipation);
		break;
	case 14:
		dissipation -= 0.001;
		printf("dissipation rate = %f\n", dissipation);
		break;
	case 15:
		vorticityConfinement = 1 - vorticityConfinement;
		vorticityConfinement ?
			printf("vorticity confinement enabled\n") :
			printf("vorticity confinement disabled\n");
		break;
	case 16:
		glutSetWindow(vortWin);
		vortVisible ? glutHideWindow() : glutShowWindow();
		vortVisible = 1 - vortVisible;
		break;
	case 17:
		g_vortForceScale += 0.01;
		printf("vorticity force scale factor = %f\n", g_vortForceScale);
		break;
	case 18:
		g_vortForceScale -= 0.01;
		printf("vorticity force scale factor = %f\n", g_vortForceScale);
		break;
	default:
		break;
	}
}

void initArrays(int dim) {
	/* allocate memory for all arrays in use */
	u = new float[dim * dim];
	v = new float[dim * dim];
	u0 = new float[dim * dim];
	v0 = new float[dim * dim];
	ub = new float[dim * dim];
	vb = new float[dim * dim];
	ua = new float[dim * dim];
	va = new float[dim * dim];
	grad_u = new float[dim * dim];
	grad_v = new float[dim * dim];
	ubt = new float[dim * dim];
	vbt = new float[dim * dim];
	uat = new float[dim * dim];
	vat = new float[dim * dim];
	grad_ut = new float[dim * dim];
	grad_vt = new float[dim * dim];
	rho = new float[dim * dim];
	rho0 = new float[dim * dim];
	u_u0 = new float[dim * dim];
	u_v0 = new float[dim * dim];
	ws = new float[dim * dim];
	w = new float[dim * dim];
	w_grad_u = new float[dim * dim];
	w_grad_v = new float[dim * dim];
	vort_force_u = new float[dim * dim];
	vort_force_v = new float[dim * dim];
	rho_s = new float[dim * dim];
	uf_s = new float[dim * dim];
	vf_s = new float[dim * dim];
	u_s = new float[dim * dim];
	v_s = new float[dim * dim];
}

int main(int argc, char **argv) {

	if (argc > 1) {
		DIM = atoi(argv[1]);
		if (DIM % 2 != 0) { DIM++; }
	}

	dim2 = DIM / 2;
	initArrays(DIM);

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);

	/* --------------VORTICITY WIN---------------------- */

	glutInitWindowSize(INITWINSIZE * 3 / 2, INITWINSIZE / 2);
	vortWin = glutCreateWindow("vorticity confinement computation");
	glutDisplayFunc(displayVortWin);
	glutReshapeFunc(reshapeVortWin);
	glutKeyboardFunc(keyboard);

	vortScalarWin = glutCreateSubWindow(vortWin,
		0, 0,
		INITWINSIZE / 3, INITWINSIZE / 2);
	glutDisplayFunc(displayScalarVort);

	vortGradWin = glutCreateSubWindow(vortWin,
		INITWINSIZE / 2, 0,
		INITWINSIZE / 3, INITWINSIZE / 2);
	glutDisplayFunc(displayGradVort);

	vortForceWin = glutCreateSubWindow(vortWin,
		INITWINSIZE, 0,
		INITWINSIZE / 3, INITWINSIZE / 2);
	glutDisplayFunc(displayForceVort);

	/* --------------VEL FIELD WIN---------------------- */

	glutInitWindowSize(INITWINSIZE * 3 / 2, INITWINSIZE);
	fieldWin = glutCreateWindow("velocity field decomposition");
	glutDisplayFunc(displayFieldWin);
	glutReshapeFunc(reshapeFieldWin);
	glutKeyboardFunc(keyboard);

	velBeforeWin = glutCreateSubWindow(fieldWin,
		0, 0,
		INITWINSIZE / 3, INITWINSIZE / 2);
	glutDisplayFunc(displayVelBefore);

	velAfterWin = glutCreateSubWindow(fieldWin,
		INITWINSIZE / 2, 0,
		INITWINSIZE / 3, INITWINSIZE / 2);
	glutDisplayFunc(displayVelAfter);

	gradientWin = glutCreateSubWindow(fieldWin,
		INITWINSIZE, 0,
		INITWINSIZE / 3, INITWINSIZE / 2);
	glutDisplayFunc(displayGradientField);

	velBeforeWinFFT = glutCreateSubWindow(fieldWin,
		0, INITWINSIZE / 2,
		INITWINSIZE / 3, INITWINSIZE / 2);
	glutDisplayFunc(displayVelBeforeFFT);

	velAfterWinFFT = glutCreateSubWindow(fieldWin,
		INITWINSIZE / 2, INITWINSIZE / 2,
		INITWINSIZE / 3, INITWINSIZE / 2);
	glutDisplayFunc(displayVelAfterFFT);

	gradientWinFFT = glutCreateSubWindow(fieldWin,
		INITWINSIZE, INITWINSIZE / 2,
		INITWINSIZE / 3, INITWINSIZE / 2);
	glutDisplayFunc(displayGradientFieldFFT);

	/* --------- MAIN WIN ------------------------------ */

	glutInitWindowSize(INITWINSIZE, INITWINSIZE);
	mainWin = glutCreateWindow("stable fluid solver");
	glutDisplayFunc(displayMainWin);
	glutReshapeFunc(reshapeMainWin);
	glutIdleFunc(updateSimulation);
	glutKeyboardFunc(keyboard);
	glutMotionFunc(drag);
	glutCreateMenu(menuFunc);
	glutAddMenuEntry("pause simulation [p]", 2);
	glutAddMenuEntry("reset simulation [r]", 4);
	glutAddMenuEntry("toggle field display [f]", 1);
	glutAddMenuEntry("toggle vorticity display [F]", 16);
	glutAddMenuEntry("toggle vorticity confinement [C]", 15);
	glutAddMenuEntry("increase vorticity force [O]", 17);
	glutAddMenuEntry("decrease vorticity force [o]", 18);
	glutAddMenuEntry("increase timestep [T]", 5);
	glutAddMenuEntry("decrease timestep [t]", 6);
	glutAddMenuEntry("increase viscosity [V]", 7);
	glutAddMenuEntry("decrease viscosity [v]", 8);
	glutAddMenuEntry("increase matter flow [S]", 11);
	glutAddMenuEntry("decrease matter flow [s]", 12);
	glutAddMenuEntry("increase dissipation rate [D]", 13);
	glutAddMenuEntry("decrease dissipation rate [d]", 14);
	glutAddMenuEntry("reset velocity [z]", 9);
	glutAddMenuEntry("reset density [a]", 10);
	glutAddMenuEntry("exit [q]", 3);
	glutAttachMenu(GLUT_RIGHT_BUTTON);

	init();

	glutMainLoop();
	return 0;
}
