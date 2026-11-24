// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-kokkos.hpp"
#include <string.h>

//------------------------------------------------------------------------------
// Backend Init
//------------------------------------------------------------------------------
static int CeedInit_Kokkos(const char *resource, Ceed ceed) {
  Ceed ceed_ref;
  char *resource_root; // from cuda ref so that device id input is possible

  CeedCallBackend(CeedGetResourceRoot(ceed, resource, ":", &resource_root));
  CeedCheck(!strcmp(resource_root, "/gpu/cuda/kokkos"), ceed, CEED_ERROR_BACKEND, "Kokkos backend cannot use resource: %s", resource);
  CeedCallBackend(CeedFree(&resource_root));

  if (!Kokkos::is_initialized()) Kokkos::initialize(); // since users can create multiple ceed objects

  // Create reference Ceed that implementation will be dispatched through unless overridden

  CeedCallBackend(CeedInit("/gpu/cuda/ref", &ceed_ref));
  CeedCallBackend(CeedSetDelegate(ceed, ceed_ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "VectorCreate", CeedVectorCreate_Kokkos));
  CeedCallBackend(CeedDestroy(&ceed_ref));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Register
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Kokkos(void) {
  return CeedRegister("/gpu/cuda/kokkos", CeedInit_Kokkos, 120);
}
//------------------------------------------------------------------------------
