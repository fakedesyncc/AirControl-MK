package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"
)

const version = "0.1.0"

type ToolStatus struct {
	Name  string `json:"name"`
	Found bool   `json:"found"`
	Path  string `json:"path,omitempty"`
}

type Report struct {
	App             string       `json:"app"`
	HelperVersion   string       `json:"helper_version"`
	CreatedAt       string       `json:"created_at"`
	OS              string       `json:"os"`
	Arch            string       `json:"arch"`
	DisplayServer   string       `json:"display_server"`
	Display         string       `json:"display,omitempty"`
	WaylandDisplay  string       `json:"wayland_display,omitempty"`
	SessionType     string       `json:"xdg_session_type,omitempty"`
	VideoDevices    []string     `json:"video_devices,omitempty"`
	Tools           []ToolStatus `json:"tools"`
	Recommendations []string     `json:"recommendations,omitempty"`
}

func main() {
	command, jsonOut, versionOut, err := parseArgs(os.Args[1:])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	if versionOut {
		fmt.Println(version)
		return
	}

	if command != "doctor" {
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", command)
		os.Exit(2)
	}

	report := buildReport()
	if jsonOut {
		encoder := json.NewEncoder(os.Stdout)
		encoder.SetIndent("", "  ")
		if err := encoder.Encode(report); err != nil {
			fmt.Fprintf(os.Stderr, "encode report: %v\n", err)
			os.Exit(1)
		}
		return
	}
	printText(report)
}

func parseArgs(args []string) (command string, jsonOut bool, versionOut bool, err error) {
	command = "doctor"
	flagArgs := make([]string, 0, len(args))
	for _, arg := range args {
		if arg == "doctor" {
			command = "doctor"
			continue
		}
		if strings.HasPrefix(arg, "-") {
			flagArgs = append(flagArgs, arg)
			continue
		}
		return "", false, false, fmt.Errorf("unknown command: %s", arg)
	}

	fs := flag.NewFlagSet("aircontrol-helper", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	fs.BoolVar(&jsonOut, "json", false, "print machine-readable JSON")
	fs.BoolVar(&versionOut, "version", false, "print helper version")
	if err := fs.Parse(flagArgs); err != nil {
		return "", false, false, err
	}
	return command, jsonOut, versionOut, nil
}

func buildReport() Report {
	report := Report{
		App:            "AirControl",
		HelperVersion:  version,
		CreatedAt:      time.Now().Format(time.RFC3339),
		OS:             runtime.GOOS,
		Arch:           runtime.GOARCH,
		DisplayServer:  displayServer(),
		Display:        os.Getenv("DISPLAY"),
		WaylandDisplay: os.Getenv("WAYLAND_DISPLAY"),
		SessionType:    os.Getenv("XDG_SESSION_TYPE"),
		VideoDevices:   videoDevices(),
		Tools:          checkTools(toolNames()),
	}
	report.Recommendations = recommendations(report)
	return report
}

func displayServer() string {
	if runtime.GOOS == "linux" {
		session := strings.ToLower(os.Getenv("XDG_SESSION_TYPE"))
		switch session {
		case "wayland", "x11":
			return session
		}
		if os.Getenv("WAYLAND_DISPLAY") != "" {
			return "wayland"
		}
		if os.Getenv("DISPLAY") != "" {
			return "x11"
		}
		return "headless"
	}
	return runtime.GOOS
}

func videoDevices() []string {
	if runtime.GOOS != "linux" {
		return nil
	}
	matches, err := filepath.Glob("/dev/video*")
	if err != nil {
		return nil
	}
	sort.Strings(matches)
	return matches
}

func toolNames() []string {
	names := []string{"flac"}
	switch runtime.GOOS {
	case "linux":
		names = append(names, "xdotool", "ydotool", "ydotoold", "wmctrl", "v4l2-ctl", "wpctl", "pactl")
	case "darwin":
		names = append(names, "osascript", "system_profiler")
	case "windows":
		names = append(names, "powershell.exe")
	}
	sort.Strings(names)
	return names
}

func checkTools(names []string) []ToolStatus {
	tools := make([]ToolStatus, 0, len(names))
	for _, name := range names {
		path, err := exec.LookPath(name)
		tools = append(tools, ToolStatus{
			Name:  name,
			Found: err == nil,
			Path:  path,
		})
	}
	return tools
}

func recommendations(report Report) []string {
	var out []string
	if report.OS == "linux" {
		if report.DisplayServer == "headless" {
			out = append(out, "No graphical Linux session detected; camera/control tests need a desktop session.")
		}
		if report.DisplayServer == "wayland" && !hasTool(report.Tools, "ydotool") {
			out = append(out, "Wayland detected; install/configure ydotoold or use an Xorg session for global input.")
		}
		if len(report.VideoDevices) == 0 {
			out = append(out, "No /dev/video* devices found; check camera permissions and the video group.")
		}
		if !hasTool(report.Tools, "xdotool") && report.DisplayServer == "x11" {
			out = append(out, "X11 session detected but xdotool is missing; install it for fallback input control.")
		}
	}
	if !hasTool(report.Tools, "flac") {
		out = append(out, "FLAC converter not found; online SpeechRecognition voice commands may be unavailable.")
	}
	return out
}

func hasTool(tools []ToolStatus, name string) bool {
	for _, tool := range tools {
		if tool.Name == name {
			return tool.Found
		}
	}
	return false
}

func printText(report Report) {
	fmt.Printf("AirControl native helper %s\n", report.HelperVersion)
	fmt.Printf("OS: %s/%s\n", report.OS, report.Arch)
	fmt.Printf("Display server: %s\n", report.DisplayServer)
	if len(report.VideoDevices) > 0 {
		fmt.Printf("Video devices: %s\n", strings.Join(report.VideoDevices, ", "))
	}
	fmt.Println("Tools:")
	for _, tool := range report.Tools {
		status := "missing"
		if tool.Found {
			status = "OK"
		}
		if tool.Path != "" {
			fmt.Printf("- %s: %s (%s)\n", tool.Name, status, tool.Path)
		} else {
			fmt.Printf("- %s: %s\n", tool.Name, status)
		}
	}
	if len(report.Recommendations) > 0 {
		fmt.Println("Recommendations:")
		for _, item := range report.Recommendations {
			fmt.Printf("- %s\n", item)
		}
	}
}
