//
// Created by aichao on 2022/4/22.
//

#ifndef FACEREC_NCNN_DEFS_H
#define FACEREC_NCNN_DEFS_H

#include <iostream>

#ifndef LITE_EXPORTS
# if (defined _WIN32 || defined WINCE || defined __CYGWIN__)
#   define LITE_EXPORTS __declspec(dllexport)
# elif defined __GNUC__ && __GNUC__ >= 4 && (defined(__APPLE__))
#   define LITE_EXPORTS __attribute__ ((visibility ("default")))
# endif
#endif

#if (defined _WIN32 || defined WINCE || defined __CYGWIN__)
# define LITE_WIN32
#elif defined __GNUC__ && __GNUC__ >= 4 && (defined(__APPLE__))
# define LITE_UNIX
#endif

#ifdef LITE_WIN32
# define NOMINMAX
#endif

#ifndef LITE_EXPORTS
# define LITE_EXPORTS
#endif

#ifndef __unused
# define __unused
#endif


#endif //FACEREC_NCNN_DEFS_H
