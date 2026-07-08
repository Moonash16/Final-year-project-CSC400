# Add project specific ProGuard rules here.
-dontwarn org.tensorflow.**
-keep class org.tensorflow.** { *; }

# Keep our model classes
-keep class com.lisive.detector.** { *; }

# Optimize for size (important for lightweight APK)
-optimizationpasses 5
-dontusemixedcaseclassnames
-dontskipnonpubliclibraryclasses
-verbose
-optimizations !code/simplification/arithmetic,!field/*,!class/merging/*

# Keep native methods
-keepclasseswithmembernames class * {
    native <methods>;
}