// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <ceed.h>
#include <ceed/backend.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

// declaring a short alias for kokkos type, so we dont have to repeat it many times
using CeedDualView_Kokkos = Kokkos::DualView<CeedScalar *>; // default view = default execution space's

typedef struct {
  CeedDualView_Kokkos *dv; // pointer to the DualView
  bool valid; // has the data been set yet ?
} CeedVector_Kokkos;

CEED_INTERN int CeedVectorCreate_Kokkos(CeedSize n, CeedVector vec);
