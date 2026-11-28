// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-kokkos.hpp"
#include <string.h>
#include <stdlib.h>

//------------------------------------------------------------------------------
// Backend Destroy
//------------------------------------------------------------------------------
static int CeedDestroy_Kokkos(Ceed ceed) {
  Ceed_Kokkos *data;

  CeedCallBackend(CeedGetData(ceed, &data));
#ifdef KOKKOS_ENABLE_CUDA
  delete data->exec;
#endif
  CeedCallBackend(CeedFree(&data));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Init
//------------------------------------------------------------------------------
static int CeedInit_Kokkos(const char *resource, Ceed ceed) {
  Ceed ceed_ref;
  char *resource_root; // to store resource name without ":device_id=..."
  Ceed_Kokkos *data;

  const char *device_spec = strstr(resource, ":device_id=");
  const int   device_id   = (device_spec) ? atoi(device_spec + 11) : -1;

  CeedCallBackend(CeedGetResourceRoot(ceed, resource, ":", &resource_root));
  CeedCheck(!strcmp(resource_root, "/gpu/cuda/kokkos"), ceed, CEED_ERROR_BACKEND, "Kokkos backend cannot use resource: %s", resource);
  CeedCallBackend(CeedFree(&resource_root));

  if (!Kokkos::is_initialized()) {
    Kokkos::InitializationSettings settings;

    if(device_id >= 0) settings.set_device_id(device_id);
    Kokkos::initialize(settings); // backend starting kokkos
    atexit(Kokkos::finalize); // backend started kokkos, so it stops kokkos at program exit
  }

#ifdef KOKKOS_ENABLE_CUDA
  else{
    CeedCheck(device_id < 0 || Kokkos::Cuda().cuda_device() == device_id, ceed, CEED_ERROR_BACKEND,
                       "Kokkos is already initialized on device %d, but resource requests device %d", Kokkos::Cuda().cuda_device(), device_id);
      }
#endif                       
  
  // Create reference Ceed that implementation will be dispatched through unless overridden
  CeedCallBackend(CeedInit("/gpu/cuda/ref", &ceed_ref));
  CeedCallBackend(CeedSetDelegate(ceed, ceed_ref));
  CeedCallBackend(CeedDestroy(&ceed_ref));

  CeedCallBackend(CeedCalloc(1, &data));
#ifdef KOKKOS_ENABLE_CUDA
  //  ToDo : use Kokkos's own stream once the full Kokkos backend works (#2065)
  data->exec = new Kokkos::Cuda((cudaStream_t)0);  // same default stream as cuda-ref
#endif
  CeedCallBackend(CeedSetData(ceed, data));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "VectorCreate", CeedVectorCreate_Kokkos));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy", CeedDestroy_Kokkos));
  return CEED_ERROR_SUCCESS;
}


//------------------------------------------------------------------------------
// Backend Register
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Kokkos(void) {
  return CeedRegister("/gpu/cuda/kokkos", CeedInit_Kokkos, 120);
}
//------------------------------------------------------------------------------
