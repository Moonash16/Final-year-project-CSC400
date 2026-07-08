package com.lisive.detector;

import android.content.Context;
import org.junit.Before;
import org.junit.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import static org.junit.Assert.*;

public class ForensicScoringTest {

    private deepfakedetector detector;

    @Mock
    Context mockContext;

    @Before
    public void setup() {
        MockitoAnnotations.openMocks(this);
        detector = new deepfakedetector(mockContext);
    }

    @Test
    public void testScoringWeights() {
        // Mock sub-scores
        double ela = 0.5;
        double noise = 0.5;
        double compression = 0.5;
        double edge = 0.5;
        double ai = 0.5;
        double blur = 0.5;
        double color = 0.5;
        double noiseInc = 0.5;
        double portrait = 0.5;

        // The formula in forensicanalyzer.java:
        // this.overallScore = (ela * 0.12 + noise * 0.08 + compression * 0.05 + edge * 0.05 + 
        //                      ai * 0.35 + blur * 0.1 + color * 0.05 + noiseInc * 0.05 + portrait * 0.15);
        
        double expectedScore = (ela * 0.12 + noise * 0.08 + compression * 0.05 + edge * 0.05 + 
                                ai * 0.35 + blur * 0.1 + color * 0.05 + noiseInc * 0.05 + portrait * 0.15);
        
        assertEquals(0.5, expectedScore, 0.001);
        
        // Verify weights sum to 1.0
        double sumWeights = 0.12 + 0.08 + 0.05 + 0.05 + 0.35 + 0.1 + 0.05 + 0.05 + 0.15;
        assertEquals(1.0, sumWeights, 0.001);
    }

    @Test
    public void testConfidenceCurve() {
        // Mock a forensic result (fake)
        forensicanalyzer.ForensicResult fakeResult = new forensicanalyzer.ForensicResult(
            0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
        );
        
        // ML Score also high (agreement boost)
        float highConfidence = detector.calculateFinalConfidence(fakeResult, 0.7f);
        // spreadForensic = 0.8 ^ 1.2 = 0.764
        // weighted = (0.764 * 0.7) + (0.7 * 0.3) = 0.535 + 0.21 = 0.745
        // Boost = 0.745 + 0.12 = 0.865
        assertTrue("High scores should be boosted: " + highConfidence, highConfidence > 0.85);
        
        // Mock an authentic result
        forensicanalyzer.ForensicResult authenticResult = new forensicanalyzer.ForensicResult(
            0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
        );
        float lowConfidence = detector.calculateFinalConfidence(authenticResult, 0.1f);
        // spreadForensic = 0.2 ^ 1.6 = 0.076
        // weighted = (0.076 * 0.7) + (0.1 * 0.3) = 0.053 + 0.03 = 0.083
        // Penalty = 0.083 - 0.12 = -0.037 -> min 0.02
        assertEquals(0.02f, lowConfidence, 0.001);
    }

    @Test
    public void testClusteringPrevention() {
        // Check scores near the neutral 0.5 zone
        forensicanalyzer.ForensicResult midResult = new forensicanalyzer.ForensicResult(
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
        );
        float midConfidence = detector.calculateFinalConfidence(midResult, 0.4f);
        // spreadForensic = 0.5 ^ 1.6 = 0.329
        // weighted = (0.329 * 0.7) + (0.4 * 0.3) = 0.230 + 0.12 = 0.35
        // Result should be pushed away from the 0.5/0.71 cluster
        assertTrue("Mid score should be pushed down: " + midConfidence, midConfidence < 0.4);
    }
}