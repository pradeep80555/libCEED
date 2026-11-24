// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-kokkos.hpp"

//------------------------------------------------------------------------------
// Has Valid Array
//------------------------------------------------------------------------------
static int CeedVectorHasValidArray_Kokkos(CeedVector vec, bool *has_valid_array){
  CeedVector_Kokkos *impl;

  CeedCallBackend(CeedVectorGetData(vec, &impl));
  *has_valid_array = impl->valid;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Destroy
//------------------------------------------------------------------------------
static int CeedVectorDestroy_Kokkos(CeedVector vec) {
  CeedVector_Kokkos *impl;

  CeedCallBackend(CeedVectorGetData(vec, &impl));
  delete impl->dv;
  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Create
//------------------------------------------------------------------------------
int CeedVectorCreate_Kokkos(CeedSize n, CeedVector vec) {
  Ceed            ceed;
  CeedVector_Kokkos *impl;

  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "HasValidArray", CeedVectorHasValidArray_Kokkos));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "Destroy", CeedVectorDestroy_Kokkos));
  CeedCallBackend(CeedDestroy(&ceed));
  CeedCallBackend(CeedCalloc(1, &impl));
  impl->dv = new CeedDualView_Kokkos("CeedVector_Kokkos", n); // valid = false from previous step (Calloc)
  CeedCallBackend(CeedVectorSetData(vec, impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------