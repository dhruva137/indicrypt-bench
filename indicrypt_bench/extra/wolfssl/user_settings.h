/* Minimal wolfCrypt configuration for IndiCrypt-Bench: every algorithm on, no OS services. */
#define WOLFCRYPT_ONLY
#define NO_FILESYSTEM
#define SINGLE_THREADED
#define NO_WRITEV
#define WOLFSSL_NO_SOCK
#define HAVE_ECC
#define HAVE_CURVE25519
#define HAVE_ED25519
#define HAVE_CURVE448
#define HAVE_ED448
#define HAVE_CHACHA
#define HAVE_POLY1305
#define HAVE_AESGCM
#define HAVE_AESCCM
#define WOLFSSL_AES_COUNTER
#define WOLFSSL_AES_DIRECT
#define WOLFSSL_SHA224
#define WOLFSSL_SHA384
#define WOLFSSL_SHA512
#define WOLFSSL_SHA3
#define HAVE_HKDF
#define WOLFSSL_KEY_GEN
#define HAVE_HASHDRBG
#define WOLFSSL_CMAC
#define HAVE_BLAKE2
#define HAVE_BLAKE2B
#define WOLFSSL_RIPEMD
#define HAVE_CAMELLIA
#define WOLFSSL_SP_MATH_ALL
#define NO_DEV_RANDOM
#define CUSTOM_RAND_GENERATE_BLOCK wc_bench_rand_block
int wc_bench_rand_block(unsigned char *out, unsigned int sz);
/* Post-quantum and stateful-hash signatures (wolfSSL's own implementations); wolfSSL marks them experimental. */
#define WOLFSSL_EXPERIMENTAL_SETTINGS
#define WOLFSSL_HAVE_KYBER
#define WOLFSSL_WC_KYBER
#define HAVE_DILITHIUM
#define WOLFSSL_WC_DILITHIUM
#define WOLFSSL_HAVE_LMS
#define WOLFSSL_WC_LMS
#define WOLFSSL_HAVE_XMSS
#define WOLFSSL_WC_XMSS
#define WOLFSSL_SHAKE128
#define WOLFSSL_SHAKE256
