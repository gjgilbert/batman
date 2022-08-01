/* The batman package: fast computation of exoplanet transit light curves
 * Copyright (C) 2015 Laura Kreidberg 
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include "numpy/arrayobject.h"

#if defined (_OPENACC) && defined(__PGI)
#  include <accelmath.h>
#else
#  include <math.h>
#endif

#if defined (_OPENMP) && !defined(_OPENACC)
#  include <omp.h>
#endif

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

static PyObject *_rsky(PyObject *self, PyObject *args);

/*static PyObject *_getf(PyObject *self, PyObject *args);*/

static PyObject *_rsky_or_f(PyObject *self, PyObject *args, int f_only)
{
    /*
        This module computes the distance between the centers of the
        star and the planet in the plane of the sky.  This parameter is
        denoted r_sky = sqrt(x^2 + y^2) in the Seager Exoplanets book
        (see the section by Murray, and Winn eq. 5).  In the Mandel & Agol
        (2002) paper, this quantity is denoted d.

        If f_only is 1, this function returns the true anomaly instead of the distance.
    */
    double tc, per, rp, b, T14, BIGD = 100.;
    int transittype, nthreads;;
    npy_intp dims[1];
    PyArrayObject *ts, *ds;
    
    if(!PyArg_ParseTuple(args,"Odddddii", &ts, &tc, &per, &rp, &b, &T14, &transittype, &nthreads)) return NULL;
    
    dims[0] = PyArray_DIMS(ts)[0];
    ds = (PyArrayObject *) PyArray_SimpleNew(1, dims, PyArray_TYPE(ts));

    double *t_array = PyArray_DATA(ts);
    double *output_array = PyArray_DATA(ds);

    const double n = 2.*M_PI/per;  // mean motion
    const double eps = 1.0e-7;

    #if defined (_OPENMP) && !defined(_OPENACC)
    omp_set_num_threads(nthreads);  //specifies number of threads (if OpenMP is supported)
    #endif

    #if defined (_OPENACC)
    #pragma acc parallel loop copyin(t_array[:dims[0]]) copyout(output_array[:dims[0]])
    #elif defined (_OPENMP)
    #pragma omp parallel for
    #endif
    for(int i = 0; i < dims[0]; i++)
    {
        double t = (t_array[i]-tc);
        double f = 2*M_PI*t/per;
        double t_phi = fmod(t,per);
        
        if (t_phi > per/2.){
            t_phi -= per;
        }

        if (f_only) {
            output_array[i] = f;
        }
        else {
            double d;
            
            //planet is nontransiting, so d is set to large value
            if (fabs(t_phi) > T14) d = BIGD;
            
            //calculates separation of centers
            else d = sqrt(b*b + 4.0/(T14*T14)*((1.0+rp)*(1.0+rp) - b*b)*(t_phi*t_phi));
            output_array[i] = d;
        }

    }
    return PyArray_Return((PyArrayObject *)ds);
}

static PyObject *_rsky(PyObject *self, PyObject *args)
{
    return _rsky_or_f(self, args, 0);
} 


static PyObject *_getf(PyObject *self, PyObject *args)
{
    return _rsky_or_f(self, args, 1);
}


static char _rsky_doc[] = """ This module computes the distance between the centers of the \
star and the planet in the plane of the sky.  This parameter is \
denoted r_sky = sqrt(x^2 + y^2) in the Seager Exoplanets book \
(see the section by Murray, and Winn eq. 5).  In the Mandel & Agol (2002) paper, \
this quantity is denoted d.\
LK 4/27/12 """;


static char _getf_doc[] = """ This module computes the true anomaly. This parameter is \
denoted f in the Seager Exoplanets book \
(see the section by Murray, and Winn eq. 44).\
BM 1/18/16 """;


static PyMethodDef _rsky_methods[] = {
  {"_rsky", _rsky,METH_VARARGS,_rsky_doc},{"_getf", _getf,METH_VARARGS,_getf_doc},{NULL}};


#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef _rsky_module = {
        PyModuleDef_HEAD_INIT,
        "_rsky",
        _rsky_doc,
        -1, 
        _rsky_methods
    };

    PyMODINIT_FUNC
    PyInit__rsky(void)
    {
        PyObject* module = PyModule_Create(&_rsky_module);
        if(!module)
        {
            return NULL;
        }
        import_array(); 
        return module;
    }
#else
    void init_rsky(void)
    {
      Py_InitModule("_rsky", _rsky_methods);
      import_array();
    }
#endif

