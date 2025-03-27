/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.h to edit this template
 */

/* 
 * File:   xtensor_lib.h
 * Author: ltsach
 *
 * Created on September 2, 2024, 10:39 AM
 */

#ifndef XTENSOR_LIB_H
#define XTENSOR_LIB_H
#include <string>
#include <sstream>
using namespace std;

#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xslice.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xindex_view.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xnpy.hpp"
#include <ctime>

typedef unsigned long ulong;
typedef xt::xarray<ulong> ulong_tensor;
typedef xt::xarray<double> double_tensor;



string shape2str(xt::svector<unsigned long> vec);
int positive_index(int idx, int size);
xt::xarray<double> outer_stack(xt::xarray<double> X, xt::xarray<double>  Y);
xt::xarray<double> diag_stack(xt::xarray<double> X);
xt::xarray<double> matmul_on_stack(xt::xarray<double> X, xt::xarray<double>  Y);


#endif /* XTENSOR_LIB_H */

