package main

import "testing"

func TestParseArgsAcceptsCommandBeforeJSONFlag(t *testing.T) {
	command, jsonOut, versionOut, err := parseArgs([]string{"doctor", "--json"})
	if err != nil {
		t.Fatalf("parseArgs returned error: %v", err)
	}
	if command != "doctor" || !jsonOut || versionOut {
		t.Fatalf("unexpected parse result: command=%q json=%v version=%v", command, jsonOut, versionOut)
	}
}

func TestParseArgsRejectsUnknownCommand(t *testing.T) {
	if _, _, _, err := parseArgs([]string{"scan"}); err == nil {
		t.Fatal("parseArgs accepted an unknown command")
	}
}

func TestLinuxWaylandRecommendations(t *testing.T) {
	report := Report{
		OS:            "linux",
		DisplayServer: "wayland",
		Tools: []ToolStatus{
			{Name: "flac", Found: true},
			{Name: "ydotool", Found: false},
		},
	}
	recs := recommendations(report)
	if len(recs) == 0 {
		t.Fatal("expected recommendations for Wayland without ydotool")
	}
	if !containsRecommendation(recs, "Wayland detected") {
		t.Fatalf("missing Wayland recommendation: %#v", recs)
	}
	if !containsRecommendation(recs, "No /dev/video* devices found") {
		t.Fatalf("missing camera device recommendation: %#v", recs)
	}
}

func containsRecommendation(items []string, needle string) bool {
	for _, item := range items {
		if len(item) >= len(needle) && item[:len(needle)] == needle {
			return true
		}
	}
	return false
}
