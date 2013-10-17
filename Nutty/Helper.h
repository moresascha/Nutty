#pragma once

#define SAFE_DELETE(a) if( (a) != NULL ) delete (a); (a) = NULL;
#define SAFE_ARRAY_DELETE(a) if( (a) != NULL ) delete[] (a); (a) = NULL;

#define TBD_FOR(__iterable) for(auto it = __iterable.begin(); it != __iterable.end(); ++it)
#define TBD_FOR_INT(intVal) for(UINT i = 0; i < intVal; ++i)

#define PrintErrorString(message) \
    { \
    std::string dest; \
    std::string t = "[CUDAH ERROR]: "; \
    std::string inLine = "\n"; \
    for(UINT i = 0; i < t.length(); ++i) \
        { \
        inLine += " "; \
        } \
        dest += t + message; \
        std::string fun = __FUNCTION__; \
        dest += inLine + "Function: " + fun; \
        std::string f = __FILE__; \
        dest += inLine  + "File: " + f; \
        char buffer[11]; \
        _itoa_s(__LINE__, buffer, 10); \
        std::string l = buffer; \
        dest += inLine + "Line: " + l; \
        dest += "\n"; \
        int id = IDABORT;\
        if(IDABORT == id)\
        {\
        DebugBreak();\
        }\
        else\
        {\
        exit(-1);\
        } \
    }

#define LOG_CRITICAL_ERROR(__text) \
    { \
    PrintErrorString(__text);\
}

#define LOG_CRITICAL_ERROR_A(__pat, __text) \
    { \
    CHAR ___b[1024]; \
    ZeroMemory(___b, 1024); \
    sprintf_s(___b, 1024, __pat, __text); \
    PrintErrorString(___b);\
}