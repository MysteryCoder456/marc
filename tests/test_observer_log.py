import unittest

from marc.agent.observation.observer import (
    AnalysisEvent,
    SurfacedContextEvent,
    _format_analysis_log,
    _format_surfaceability_log,
)


class ObserverLogFormattingTests(unittest.TestCase):
    def test_surfaceability_log_preserves_mixed_event_order(self):
        events = [
            AnalysisEvent("Opened the implementation plan."),
            SurfacedContextEvent("The prior decision was to use one log."),
            AnalysisEvent("Started updating the observer."),
        ]

        self.assertEqual(
            _format_surfaceability_log(events),
            "1. ANALYSIS: Opened the implementation plan.\n"
            "2. SURFACED CONTEXT: The prior decision was to use one log.\n"
            "3. ANALYSIS: Started updating the observer.",
        )

    def test_analysis_log_filters_surfaced_context_and_renumbers(self):
        events = [
            AnalysisEvent("Opened the implementation plan."),
            SurfacedContextEvent("The prior decision was to use one log."),
            AnalysisEvent("Started updating the observer."),
        ]

        self.assertEqual(
            _format_analysis_log(events),
            "1. Opened the implementation plan.\n"
            "2. Started updating the observer.",
        )


if __name__ == "__main__":
    unittest.main()
