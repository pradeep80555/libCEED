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
// Vector Set Array (here is some data, and put it in the vector)
//------------------------------------------------------------------------------
static int CeedVectorSetArray_Kokkos(CeedVector vec, CeedMemType mem_type, CeedCopyMode copy_mode, CeedScalar *array) {
  CeedVector_Kokkos *impl;
  CeedSize length;

  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedCallBackend(CeedVectorGetLength(vec, &length));

  // Temporary: USE_POINTER (step 3b) and OWN_POINTER (step 3c) come later
  CeedCheck(copy_mode == CEED_COPY_VALUES, CeedVectorReturnCeed(vec), CEED_ERROR_UNSUPPORTED,
            "Only CEED_COPY_VALUES is supported so far");
  
  // new data replaces everything, therefore neither side (host or device) is newer anymore
  impl->dv->clear_sync_state();

  switch (mem_type) {
    case CEED_MEM_HOST: {
      if (array) {
        CeedDualView_Kokkos::t_host_um user_view(array, (size_t)length); // wrapping the given array into unmanaged host view
        Kokkos::deep_copy(impl->dv->view_host(), user_view); // copying user_view into host part of the impl DualView
      }
      impl->dv->modify_host();
      break;
    }
    case CEED_MEM_DEVICE: {
      if (array) {
        CeedDualView_Kokkos::t_dev_um user_view(array, (size_t)length); // wrapping the given array into unmanaged device view
        Kokkos::deep_copy(impl->dv->view_device(), user_view); // copying user_view into device part of the impl DualView
      }
      impl->dv->modify_device();
      break;
    }
  }
  impl->valid = true; // was false by default
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
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "SetArray", CeedVectorSetArray_Kokkos));
  CeedCallBackend(CeedDestroy(&ceed));
  CeedCallBackend(CeedCalloc(1, &impl));
  impl->dv = new CeedDualView_Kokkos("CeedVector_Kokkos", n); // valid = false from previous step (Calloc)
  CeedCallBackend(CeedVectorSetData(vec, impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------