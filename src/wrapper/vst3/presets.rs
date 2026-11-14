//! Factory and user preset discovery, `.vstpreset` file reading and writing, and program list
//! support for the VST3 wrapper.
//!
//! The state stored inside a `.vstpreset` file's component state chunk is the exact same
//! (optionally Zstandard-compressed) JSON blob produced by
//! [`serialize_json()`][crate::wrapper::state::serialize_json()], which is also what the CLAP
//! wrapper stores through its `clap_plugin_state` implementation. Presets can thus be moved
//! between the VST3 and CLAP versions of a plugin, and [`Plugin::filter_state()`] based migrations
//! apply to presets in the same way as they apply to host-restored state.
//!
//! None of the functions in this module may be called from the audio thread. The preset directory
//! scan happens once when the wrapper is created (on the main thread), and preset files are only
//! read and written from the main/GUI thread.
//!
//! <https://steinbergmedia.github.io/vst3_dev_portal/pages/Technical+Documentation/Locations+Format/Preset+Locations.html>

use anyhow::{anyhow, bail, Context, Result};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::prelude::Plugin;
use crate::wrapper::state::{self, PluginState};

/// The maximum size of a single `.vstpreset` file. Files larger than this are rejected before they
/// are read into memory.
pub(crate) const MAX_PRESET_FILE_SIZE: u64 = 32 * 1024 * 1024;
/// The maximum number of presets that will be scanned. This puts an upper bound on the amount of
/// memory the program list can take up.
pub(crate) const MAX_PRESET_COUNT: usize = 2048;
/// The maximum length of a preset name in Unicode characters.
pub(crate) const MAX_PRESET_NAME_LEN: usize = 128;
/// The maximum directory depth used when scanning preset directories for `.vstpreset` files.
const MAX_SCAN_DEPTH: usize = 4;
/// The file extension used for VST3 presets, without the leading dot.
pub(crate) const PRESET_FILE_EXTENSION: &str = "vstpreset";

/// The ID of the program list exposed through `IUnitInfo` when the plugin has any presets.
pub(crate) const VST3_PROGRAM_LIST_ID: i32 = 1;

// The `.vstpreset` file format constants. The format consists of a header, followed by the raw
// chunk data, followed by a chunk list at the offset stored in the header. See `PresetFile` in the
// VST3 SDK for the reference implementation.
const PRESET_MAGIC: &[u8; 5] = b"VST3\0";
const PRESET_FORMAT_VERSION: i32 = 1;
/// The chunk ID for the component (`IComponent`) state. This is the only chunk NIH-plug writes,
/// and the only one it reads.
const CHUNK_ID_COMPONENT_STATE: &[u8; 4] = b"Comp";
/// The class ID is stored as a 32 character ASCII hex string.
const CLASS_ID_STRING_LEN: usize = 32;
const HEADER_SIZE: usize = PRESET_MAGIC.len() + 4 + CLASS_ID_STRING_LEN + 8;
const CHUNK_ENTRY_SIZE: usize = 4 + 8 + 8;
/// Sanity limit on the number of chunks in a preset file. Regular presets contain one or two
/// chunks (the component state and optionally the controller state).
const MAX_CHUNK_COUNT: i32 = 16;

/// Where a preset came from. Factory presets are shipped with the plugin and are read-only, user
/// presets are written to the user's preset directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PresetOrigin {
    Factory,
    User,
}

/// A preset discovered in one of the preset directories.
#[derive(Debug, Clone)]
pub(crate) struct Preset {
    /// The preset's name, derived from the file name.
    pub name: String,
    pub origin: PresetOrigin,
    pub path: PathBuf,
}

/// The presets discovered in the factory and user preset directories. This is scanned once when
/// the wrapper is created on the main thread, and appended to when the user saves a new preset.
#[derive(Debug, Default)]
pub(crate) struct PresetLibrary {
    /// All discovered presets, sorted by origin (factory first) and then by name.
    presets: Vec<Preset>,
}

impl PresetLibrary {
    /// Scan the given `(directory, origin)` pairs for `.vstpreset` files. Missing or unreadable
    /// directories are skipped. The number of presets is capped at [`MAX_PRESET_COUNT`].
    pub fn scan(directories: &[(PathBuf, PresetOrigin)]) -> Self {
        Self::scan_with_limit(directories, MAX_PRESET_COUNT)
    }

    /// [`Self::scan()`] with an explicit preset limit, used for testing.
    fn scan_with_limit(directories: &[(PathBuf, PresetOrigin)], max_presets: usize) -> Self {
        let mut presets = Vec::new();
        for (directory, origin) in directories {
            if presets.len() >= max_presets {
                break;
            }
            let mut origin_presets = Vec::new();
            scan_directory(
                directory,
                *origin,
                0,
                max_presets,
                &mut origin_presets,
            );
            // Sort each origin's presets by name so the program list order is stable
            origin_presets.sort_by(|a, b| {
                a.name
                    .to_lowercase()
                    .cmp(&b.name.to_lowercase())
                    .then_with(|| a.path.cmp(&b.path))
            });
            presets.append(&mut origin_presets);
        }

        Self { presets }
    }

    pub fn presets(&self) -> &[Preset] {
        &self.presets
    }

    /// Append a preset to the library. Used after saving a new user preset. The preset is appended
    /// at the end so the indices of the existing presets (which the host may already know about)
    /// don't change.
    pub fn add(&mut self, preset: Preset) {
        self.presets.push(preset);
    }
}

/// Recursively scan a directory for `.vstpreset` files. Symlinks are not followed, so this cannot
/// loop.
fn scan_directory(
    directory: &Path,
    origin: PresetOrigin,
    depth: usize,
    max_presets: usize,
    presets: &mut Vec<Preset>,
) {
    if depth > MAX_SCAN_DEPTH || presets.len() >= max_presets {
        return;
    }

    let entries = match fs::read_dir(directory) {
        Ok(entries) => entries,
        // A missing or unreadable preset directory is not an error, it just means there are no
        // presets there
        Err(_) => return,
    };
    // Sort the entries so the scan order (and thus the program list order) is deterministic
    let mut entries: Vec<_> = entries.filter_map(|entry| entry.ok()).collect();
    entries.sort_by_key(|entry| entry.file_name());

    for entry in entries {
        if presets.len() >= max_presets {
            break;
        }

        let path = entry.path();
        let file_type = match entry.file_type() {
            Ok(file_type) => file_type,
            Err(_) => continue,
        };
        if file_type.is_dir() {
            scan_directory(&path, origin, depth + 1, max_presets, presets);
        } else if file_type.is_file()
            && path
                .extension()
                .map_or(false, |ext| ext.eq_ignore_ascii_case(PRESET_FILE_EXTENSION))
        {
            let name = match path.file_stem() {
                Some(stem) => stem.to_string_lossy().into_owned(),
                None => continue,
            };
            if name.is_empty() {
                continue;
            }

            presets.push(Preset { name, origin, path });
        }
    }
}

/// The directories scanned for presets, as `(directory, origin)` pairs. Factory directories come
/// first so factory presets appear before user presets in the program list.
pub(crate) fn preset_directories<P: Plugin>() -> Vec<(PathBuf, PresetOrigin)> {
    let mut directories: Vec<(PathBuf, PresetOrigin)> = factory_preset_directories::<P>()
        .into_iter()
        .map(|dir| (dir, PresetOrigin::Factory))
        .collect();
    if let Some(user_dir) = user_preset_directory::<P>() {
        directories.push((user_dir, PresetOrigin::User));
    }

    directories
}

/// The directory user presets are saved to and loaded from, if the platform's user directory could
/// be determined.
pub(crate) fn user_preset_directory<P: Plugin>() -> Option<PathBuf> {
    #[cfg(all(target_family = "unix", not(target_os = "macos")))]
    let base = std::env::var_os("HOME").map(PathBuf::from).map(|home| home.join(".vst3"));
    #[cfg(target_os = "macos")]
    let base = std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join("Library").join("Audio").join("Presets"));
    #[cfg(target_os = "windows")]
    let base = std::env::var_os("USERPROFILE").map(PathBuf::from).map(|home| {
        home.join("Documents").join("VST3 Presets")
    });

    base.map(|base| base.join(P::VENDOR).join(P::NAME))
}

/// The system-wide directories scanned for factory presets.
fn factory_preset_directories<P: Plugin>() -> Vec<PathBuf> {
    #[cfg(all(target_family = "unix", not(target_os = "macos")))]
    let bases = [
        PathBuf::from("/usr/share/vst3/presets"),
        PathBuf::from("/usr/local/share/vst3/presets"),
    ];
    #[cfg(target_os = "macos")]
    let bases = [PathBuf::from("/Library/Audio/Presets")];
    #[cfg(target_os = "windows")]
    let bases = [std::env::var_os("PROGRAMDATA")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\ProgramData"))
        .join("VST3 Presets")];

    bases
        .into_iter()
        .map(|base| base.join(P::VENDOR).join(P::NAME))
        .collect()
}

/// Check a preset name for use as a file name within the user preset directory. This rejects
/// anything that could escape the preset directory (path separators, `..`, ...), names that are
/// invalid on any of the supported platforms, and names that are unreasonably long. Returns the
/// trimmed name on success.
pub(crate) fn sanitize_preset_name(name: &str) -> Result<String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        bail!("The preset name cannot be empty");
    }
    if trimmed.chars().count() > MAX_PRESET_NAME_LEN {
        bail!(
            "The preset name cannot be longer than {} characters",
            MAX_PRESET_NAME_LEN
        );
    }
    // Reject anything that could be used for path traversal. Since path separators are rejected
    // altogether, joining the sanitized name to the user preset directory can never escape it
    if trimmed == "." || trimmed == ".." {
        bail!("Invalid preset name {:?}", trimmed);
    }
    for ch in trimmed.chars() {
        match ch {
            // Characters that are either path separators or that are not allowed in file names on
            // one of the supported platforms
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => {
                bail!("The preset name contains an invalid character {:?}", ch)
            }
            ch if ch.is_control() => {
                bail!("The preset name contains a control character")
            }
            _ => (),
        }
    }
    // Trailing dots are not allowed in file names on Windows
    if trimmed.ends_with('.') {
        bail!("The preset name cannot end with a dot");
    }

    Ok(trimmed.to_string())
}

/// Convert a 16 byte VST3 class ID (in platform byte order, i.e. the order the TUID is stored in
/// memory) to the 32 character hex string used in the `.vstpreset` header. This matches
/// `UIDtoString()` from the VST3 SDK: the ID is interpreted as four native-endian 32-bit integers
/// which are formatted as eight digit uppercase hex numbers.
fn class_id_to_preset_string(class_id: &[u8; 16]) -> String {
    let mut string = String::with_capacity(CLASS_ID_STRING_LEN);
    for word in class_id.chunks_exact(4) {
        let value = u32::from_le_bytes(word.try_into().unwrap());
        string.push_str(&format!("{value:08X}"));
    }
    string
}

/// The inverse of [`class_id_to_preset_string()`]. Accepts both upper and lowercase hex strings.
fn class_id_from_preset_string(string: &str) -> Option<[u8; 16]> {
    if string.len() != CLASS_ID_STRING_LEN {
        return None;
    }

    let mut class_id = [0u8; 16];
    for (word_idx, word) in string.as_bytes().chunks_exact(4 * 2).enumerate() {
        let word_str = std::str::from_utf8(word).ok()?;
        let value = u32::from_str_radix(word_str, 16).ok()?;
        class_id[word_idx * 4..][..4].copy_from_slice(&value.to_le_bytes());
    }

    Some(class_id)
}

/// Serialize a component state blob (as produced by
/// [`serialize_json()`][crate::wrapper::state::serialize_json()]) to the `.vstpreset` file format.
pub(crate) fn serialize_vstpreset(class_id: &[u8; 16], component_state: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(HEADER_SIZE + component_state.len() + 4 + CHUNK_ENTRY_SIZE);

    // Header
    result.extend_from_slice(PRESET_MAGIC);
    result.extend_from_slice(&PRESET_FORMAT_VERSION.to_le_bytes());
    result.extend_from_slice(class_id_to_preset_string(class_id).as_bytes());
    let list_offset_position = result.len();
    // Patched with the chunk list's offset below
    result.extend_from_slice(&[0u8; 8]);

    // Chunk data
    let component_state_offset = result.len() as i64;
    result.extend_from_slice(component_state);

    // Chunk list
    let list_offset = result.len() as i64;
    result.extend_from_slice(&1i32.to_le_bytes());
    result.extend_from_slice(CHUNK_ID_COMPONENT_STATE);
    result.extend_from_slice(&component_state_offset.to_le_bytes());
    result.extend_from_slice(&(component_state.len() as i64).to_le_bytes());

    result[list_offset_position..][..8].copy_from_slice(&list_offset.to_le_bytes());

    result
}

/// Parse a `.vstpreset` file and return the component state chunk. The class ID stored in the
/// preset must match `expected_class_id`. Returns an error and does not modify anything if the
/// file is malformed, truncated, or belongs to a different plugin.
pub(crate) fn parse_vstpreset<'a>(data: &'a [u8], expected_class_id: &[u8; 16]) -> Result<&'a [u8]> {
    if data.len() > MAX_PRESET_FILE_SIZE as usize {
        bail!(
            "The preset file is too large ({} bytes)",
            data.len()
        );
    }
    if data.len() < HEADER_SIZE {
        bail!("The preset file is truncated ({} bytes)", data.len());
    }
    if data[..PRESET_MAGIC.len()] != PRESET_MAGIC[..] {
        bail!("The preset file does not start with the VST3 preset magic");
    }

    let version = i32::from_le_bytes(data[5..9].try_into().unwrap());
    if version != PRESET_FORMAT_VERSION {
        bail!("Unsupported preset format version {version}");
    }

    let class_id_string = std::str::from_utf8(&data[9..9 + CLASS_ID_STRING_LEN])
        .context("The preset file contains an invalid class ID")?;
    let class_id = class_id_from_preset_string(class_id_string)
        .ok_or_else(|| anyhow!("The preset file contains an invalid class ID"))?;
    if &class_id != expected_class_id {
        bail!("The preset belongs to a different plugin");
    }

    let list_offset = i64::from_le_bytes(data[41..49].try_into().unwrap());
    if list_offset < HEADER_SIZE as i64 || list_offset + 4 > data.len() as i64 {
        bail!("The preset file contains an invalid chunk list offset");
    }
    let list_offset = list_offset as usize;

    let chunk_count = i32::from_le_bytes(data[list_offset..][..4].try_into().unwrap());
    if chunk_count < 0 || chunk_count > MAX_CHUNK_COUNT {
        bail!("The preset file contains an invalid chunk count");
    }
    let chunk_list_size = 4 + chunk_count as usize * CHUNK_ENTRY_SIZE;
    if list_offset + chunk_list_size > data.len() {
        bail!("The preset file contains a truncated chunk list");
    }

    // Find the component state chunk. Other chunks (like the controller state chunk or metadata)
    // are ignored
    for chunk_idx in 0..chunk_count as usize {
        let entry_offset = list_offset + 4 + chunk_idx * CHUNK_ENTRY_SIZE;
        let chunk_id = &data[entry_offset..][..4];
        let chunk_offset = i64::from_le_bytes(data[entry_offset + 4..][..8].try_into().unwrap());
        let chunk_size = i64::from_le_bytes(data[entry_offset + 12..][..8].try_into().unwrap());

        if chunk_id == CHUNK_ID_COMPONENT_STATE {
            if chunk_offset < 0
                || chunk_size < 0
                || chunk_offset + chunk_size > data.len() as i64
            {
                bail!("The preset file contains an invalid component state chunk");
            }
            if chunk_size as usize > state::MAX_STATE_BYTES {
                bail!("The preset's state is too large ({} bytes)", chunk_size);
            }

            return Ok(&data[chunk_offset as usize..][..chunk_size as usize]);
        }
    }

    bail!("The preset file does not contain a component state chunk")
}

/// Read and parse a `.vstpreset` file. The file is only read into memory after checking its size,
/// and the read is capped at [`MAX_PRESET_FILE_SIZE`] bytes. Returns an error without touching the
/// plugin's current state if anything goes wrong.
pub(crate) fn load_preset_file(path: &Path, expected_class_id: &[u8; 16]) -> Result<PluginState> {
    let metadata = fs::metadata(path)
        .with_context(|| format!("Could not read the preset file '{}'", path.display()))?;
    if metadata.len() > MAX_PRESET_FILE_SIZE {
        bail!(
            "The preset file '{}' is too large ({} bytes)",
            path.display(),
            metadata.len()
        );
    }

    let file = fs::File::open(path)
        .with_context(|| format!("Could not open the preset file '{}'", path.display()))?;
    let mut data = Vec::with_capacity(metadata.len() as usize);
    file.take(MAX_PRESET_FILE_SIZE + 1)
        .read_to_end(&mut data)
        .with_context(|| format!("Could not read the preset file '{}'", path.display()))?;
    if data.len() as u64 > MAX_PRESET_FILE_SIZE {
        bail!("The preset file '{}' is too large", path.display());
    }

    let state_bytes = parse_vstpreset(&data, expected_class_id)
        .with_context(|| format!("Could not parse the preset file '{}'", path.display()))?;

    // SAFETY: This function doesn't dereference any pointers, it's marked unsafe for consistency
    // with the other state functions
    unsafe { state::deserialize_json(state_bytes) }
        .ok_or_else(|| anyhow!("Could not deserialize the preset file '{}'", path.display()))
}

/// Write a state blob to a `.vstpreset` file. The file is first written to a temporary file in the
/// same directory and then renamed over the target, so a concurrent reader (or a crash) never
/// observes a partially written preset. Callers should serialize concurrent saves to the same path
/// (see `WrapperInner::save_user_preset()`).
pub(crate) fn save_preset_file(
    path: &Path,
    class_id: &[u8; 16],
    state_bytes: &[u8],
) -> Result<()> {
    if state_bytes.len() > state::MAX_STATE_BYTES {
        bail!(
            "The plugin state is too large to save ({} bytes)",
            state_bytes.len()
        );
    }
    let data = serialize_vstpreset(class_id, state_bytes);

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "Could not create the preset directory '{}'",
                parent.display()
            )
        })?;
    }

    // Write to a unique temporary file in the same directory first so the rename below is atomic
    let file_name = path
        .file_name()
        .ok_or_else(|| anyhow!("Invalid preset path '{}'", path.display()))?
        .to_string_lossy();
    let temp_path = path.with_file_name(format!(
        ".{file_name}.tmp-{}",
        std::process::id()
    ));
    // Clean up the temporary file if any of the writes below fail
    let write_result = (|| -> Result<()> {
        fs::write(&temp_path, &data)
            .with_context(|| format!("Could not write to '{}'", temp_path.display()))?;
        fs::rename(&temp_path, path).with_context(|| {
            format!(
                "Could not rename '{}' to '{}'",
                temp_path.display(),
                path.display()
            )
        })?;
        Ok(())
    })();
    if write_result.is_err() {
        let _ = fs::remove_file(&temp_path);
    }
    write_result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::internals::ParamPtr;
    use crate::params::{BoolParam, FloatParam, FloatRange, Param, Params};
    use crate::prelude::{
        AuxiliaryBuffers, Buffer, ProcessContext, ProcessStatus,
    };
    use crate::wrapper::state::{self, PluginState};
    use parking_lot::Mutex;
    use std::collections::{BTreeMap, HashMap};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    const TEST_CLASS_ID: [u8; 16] = *b"nih-plug-test-id";
    const OTHER_CLASS_ID: [u8; 16] = *b"other-plugin-id!";

    /// A temporary directory that is removed when dropped.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(test_name: &str) -> Self {
            static COUNTER: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "nih-plug-preset-test-{}-{}-{}",
                test_name,
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::SeqCst)
            ));
            fs::create_dir_all(&path).unwrap();
            TempDir(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// A minimal set of parameters used to test state (de)serialization. Contains a persisted
    /// non-parameter field to simulate things like sample data references.
    struct TestParams {
        gain: FloatParam,
        enable: BoolParam,
        sample_path: Mutex<String>,
    }

    impl Default for TestParams {
        fn default() -> Self {
            Self {
                gain: FloatParam::new("Gain", 0.5, FloatRange::Linear { min: 0.0, max: 1.0 }),
                enable: BoolParam::new("Enable", true),
                sample_path: Mutex::new(String::new()),
            }
        }
    }

    unsafe impl Params for TestParams {
        fn param_map(&self) -> Vec<(String, ParamPtr, String)> {
            vec![
                (
                    "gain".to_string(),
                    ParamPtr::FloatParam(&self.gain),
                    String::new(),
                ),
                (
                    "enable".to_string(),
                    ParamPtr::BoolParam(&self.enable),
                    String::new(),
                ),
            ]
        }

        fn serialize_fields(&self) -> BTreeMap<String, String> {
            let mut fields = BTreeMap::new();
            fields.insert(
                "sample_path".to_string(),
                serde_json::to_string(&*self.sample_path.lock()).unwrap(),
            );
            fields
        }

        fn deserialize_fields(&self, serialized: &BTreeMap<String, String>) {
            // A missing field (e.g. because the preset predates the field, or because the sample
            // is missing) is not an error, the current value is kept
            if let Some(data) = serialized.get("sample_path") {
                if let Ok(value) = serde_json::from_str::<String>(data) {
                    *self.sample_path.lock() = value;
                }
            }
        }
    }

    /// A minimal plugin implementation used to test state (de)serialization, including
    /// `filter_state()` based version migrations.
    struct TestPlugin {
        params: Arc<TestParams>,
    }

    impl Default for TestPlugin {
        fn default() -> Self {
            Self {
                params: Arc::new(TestParams::default()),
            }
        }
    }

    impl Plugin for TestPlugin {
        const NAME: &'static str = "Preset Test";
        const VENDOR: &'static str = "NIH-plug";
        const URL: &'static str = "https://example.com";
        const EMAIL: &'static str = "test@example.com";
        const VERSION: &'static str = "2.0.0";

        const AUDIO_IO_LAYOUTS: &'static [crate::prelude::AudioIOLayout] = &[];

        type SysExMessage = ();
        type BackgroundTask = ();

        fn params(&self) -> Arc<dyn Params> {
            self.params.clone()
        }

        fn filter_state(state: &mut PluginState) {
            // Simulate a migration from version 1.x, where the `gain` parameter was still called
            // `old_gain`
            if state.version.starts_with("1.") {
                if let Some(value) = state.params.remove("old_gain") {
                    state.params.insert("gain".to_string(), value);
                }
                state.version = "2.0.0".to_string();
            }
        }

        fn process(
            &mut self,
            _buffer: &mut Buffer,
            _aux: &mut AuxiliaryBuffers,
            _context: &mut impl ProcessContext<Self>,
        ) -> ProcessStatus {
            ProcessStatus::Normal
        }
    }

    /// Build the lookup tables the wrappers normally build, for use with the `state` functions.
    fn make_param_tables(
        params: &Arc<TestParams>,
    ) -> (HashMap<u32, ParamPtr>, HashMap<String, u32>) {
        let mut param_by_hash = HashMap::new();
        let mut param_id_to_hash = HashMap::new();
        for (id, ptr, _) in params.param_map() {
            let hash = crate::wrapper::util::hash_param_id(&id);
            param_by_hash.insert(hash, ptr);
            param_id_to_hash.insert(id, hash);
        }
        (param_by_hash, param_id_to_hash)
    }

    fn serialize_test_state(params: &Arc<TestParams>) -> Vec<u8> {
        let (param_by_hash, param_id_to_hash) = make_param_tables(params);
        unsafe {
            state::serialize_json::<TestPlugin>(
                params.clone(),
                state::make_params_iter(&param_by_hash, &param_id_to_hash),
            )
        }
        .unwrap()
    }

    fn deserialize_test_state(params: &Arc<TestParams>, state: &mut PluginState) -> bool {
        let (param_by_hash, param_id_to_hash) = make_param_tables(params);
        unsafe {
            state::deserialize_object::<TestPlugin>(
                state,
                params.clone(),
                state::make_params_getter(&param_by_hash, &param_id_to_hash),
                None,
            )
        }
    }

    #[test]
    fn class_id_string_roundtrip() {
        let string = class_id_to_preset_string(&TEST_CLASS_ID);
        assert_eq!(string.len(), CLASS_ID_STRING_LEN);
        assert_eq!(
            class_id_from_preset_string(&string),
            Some(TEST_CLASS_ID),
            "The class ID string should round-trip"
        );

        // The parser should also accept lowercase hex
        assert_eq!(
            class_id_from_preset_string(&string.to_lowercase()),
            Some(TEST_CLASS_ID)
        );

        assert_eq!(class_id_from_preset_string("too short"), None);
        assert_eq!(class_id_from_preset_string(&"z".repeat(32)), None);
    }

    #[test]
    fn vstpreset_roundtrip() {
        let state = b"{\"version\":\"2.0.0\",\"params\":{},\"fields\":{}}";
        let data = serialize_vstpreset(&TEST_CLASS_ID, state);

        // The file should start with the magic, the version, and the ASCII class ID
        assert_eq!(&data[..5], PRESET_MAGIC);
        assert_eq!(i32::from_le_bytes(data[5..9].try_into().unwrap()), 1);
        assert_eq!(
            std::str::from_utf8(&data[9..41]).unwrap(),
            class_id_to_preset_string(&TEST_CLASS_ID)
        );

        let parsed = parse_vstpreset(&data, &TEST_CLASS_ID).unwrap();
        assert_eq!(parsed, state);
    }

    #[test]
    fn vstpreset_parse_rejects_bad_files() {
        let state = b"{}";
        let valid = serialize_vstpreset(&TEST_CLASS_ID, state);

        // Bad magic
        let mut bad = valid.clone();
        bad[0] = b'X';
        assert!(parse_vstpreset(&bad, &TEST_CLASS_ID).is_err());

        // Unsupported version
        let mut bad = valid.clone();
        bad[5..9].copy_from_slice(&2i32.to_le_bytes());
        assert!(parse_vstpreset(&bad, &TEST_CLASS_ID).is_err());

        // Class ID from another plugin
        assert!(parse_vstpreset(&valid, &OTHER_CLASS_ID).is_err());

        // Truncated header
        assert!(parse_vstpreset(&valid[..20], &TEST_CLASS_ID).is_err());
        // Empty file
        assert!(parse_vstpreset(&[], &TEST_CLASS_ID).is_err());

        // Chunk list offset pointing out of bounds
        let mut bad = valid.clone();
        bad[41..49].copy_from_slice(&(valid.len() as i64 + 1000).to_le_bytes());
        assert!(parse_vstpreset(&bad, &TEST_CLASS_ID).is_err());

        // Negative chunk list offset
        let mut bad = valid.clone();
        bad[41..49].copy_from_slice(&(-1i64).to_le_bytes());
        assert!(parse_vstpreset(&bad, &TEST_CLASS_ID).is_err());

        // Chunk data pointing out of bounds. The chunk list starts right after the state data.
        let list_offset = i64::from_le_bytes(valid[41..49].try_into().unwrap()) as usize;
        let mut bad = valid.clone();
        bad[list_offset + 4 + 4..][..8].copy_from_slice(&(valid.len() as i64).to_le_bytes());
        assert!(parse_vstpreset(&bad, &TEST_CLASS_ID).is_err());

        // Absurd chunk count
        let mut bad = valid.clone();
        bad[list_offset..][..4].copy_from_slice(&1000i32.to_le_bytes());
        assert!(parse_vstpreset(&bad, &TEST_CLASS_ID).is_err());

        // Garbage data
        assert!(parse_vstpreset(&[0xAB; 4096], &TEST_CLASS_ID).is_err());
    }

    #[test]
    fn sanitize_preset_name_accepts_normal_names() {
        assert_eq!(sanitize_preset_name("Init").unwrap(), "Init");
        assert_eq!(sanitize_preset_name("  Padded  ").unwrap(), "Padded");
        assert_eq!(
            sanitize_preset_name("Ünïcödé プリセット 🎛").unwrap(),
            "Ünïcödé プリセット 🎛"
        );
    }

    #[test]
    fn sanitize_preset_name_rejects_traversal_and_invalid_names() {
        // Path traversal
        assert!(sanitize_preset_name("..").is_err());
        assert!(sanitize_preset_name(".").is_err());
        assert!(sanitize_preset_name("../evil").is_err());
        assert!(sanitize_preset_name("..\\evil").is_err());
        assert!(sanitize_preset_name("foo/bar").is_err());
        assert!(sanitize_preset_name("foo\\bar").is_err());
        assert!(sanitize_preset_name("/absolute").is_err());

        // Empty and overly long names
        assert!(sanitize_preset_name("").is_err());
        assert!(sanitize_preset_name("   ").is_err());
        assert!(sanitize_preset_name(&"a".repeat(MAX_PRESET_NAME_LEN + 1)).is_err());
        assert!(sanitize_preset_name(&"a".repeat(MAX_PRESET_NAME_LEN)).is_ok());

        // Characters that are invalid in file names
        assert!(sanitize_preset_name("a:b").is_err());
        assert!(sanitize_preset_name("a*b").is_err());
        assert!(sanitize_preset_name("a?b").is_err());
        assert!(sanitize_preset_name("a\"b").is_err());
        assert!(sanitize_preset_name("a<b").is_err());
        assert!(sanitize_preset_name("a>b").is_err());
        assert!(sanitize_preset_name("a|b").is_err());
        assert!(sanitize_preset_name("a\0b").is_err());
        assert!(sanitize_preset_name("a\nb").is_err());

        // Trailing dots are invalid on Windows
        assert!(sanitize_preset_name("name.").is_err());
    }

    #[test]
    fn scan_finds_factory_and_user_presets() {
        let temp = TempDir::new("scan");
        let factory_dir = temp.path().join("factory");
        let user_dir = temp.path().join("user");
        fs::create_dir_all(factory_dir.join("Bass")).unwrap();
        fs::create_dir_all(&user_dir).unwrap();

        fs::write(factory_dir.join("Init.vstpreset"), b"factory").unwrap();
        fs::write(factory_dir.join("Bass").join("Deep.vstpreset"), b"nested").unwrap();
        fs::write(user_dir.join("My Preset.vstpreset"), b"user").unwrap();
        // Non-preset files should be ignored
        fs::write(user_dir.join("notes.txt"), b"not a preset").unwrap();
        fs::write(user_dir.join(".vstpreset"), b"no name").unwrap();

        let library = PresetLibrary::scan(&[
            (factory_dir, PresetOrigin::Factory),
            (user_dir, PresetOrigin::User),
            // Missing directories should be silently skipped
            (temp.path().join("does-not-exist"), PresetOrigin::User),
        ]);

        let presets = library.presets();
        assert_eq!(presets.len(), 3);
        // Factory presets come first, sorted by name within each origin
        assert_eq!(presets[0].name, "Deep");
        assert_eq!(presets[0].origin, PresetOrigin::Factory);
        assert_eq!(presets[1].name, "Init");
        assert_eq!(presets[1].origin, PresetOrigin::Factory);
        assert_eq!(presets[2].name, "My Preset");
        assert_eq!(presets[2].origin, PresetOrigin::User);
    }

    #[test]
    fn scan_respects_preset_limit() {
        let temp = TempDir::new("scan-limit");
        let dir = temp.path().join("presets");
        fs::create_dir_all(&dir).unwrap();
        for i in 0..10 {
            fs::write(dir.join(format!("preset-{i:02}.vstpreset")), b"x").unwrap();
        }

        let library = PresetLibrary::scan_with_limit(&[(dir, PresetOrigin::User)], 4);
        assert_eq!(library.presets().len(), 4);
    }

    #[test]
    fn preset_file_roundtrip_restores_state() {
        let temp = TempDir::new("roundtrip");

        // Serialize a state with non-default values and a persisted field
        let source_params = Arc::new(TestParams::default());
        source_params.gain.set_plain_value(0.75);
        source_params.enable.set_plain_value(false);
        *source_params.sample_path.lock() = "/samples/kick.wav".to_string();
        let state_bytes = serialize_test_state(&source_params);

        // Write it to a preset file and load it back
        let preset_path = temp.path().join("user").join("Test.vstpreset");
        save_preset_file(&preset_path, &TEST_CLASS_ID, &state_bytes).unwrap();
        let mut loaded_state = load_preset_file(&preset_path, &TEST_CLASS_ID).unwrap();

        // The version should have been stored with the state
        assert_eq!(loaded_state.version, "2.0.0");

        let target_params = Arc::new(TestParams::default());
        assert!(deserialize_test_state(&target_params, &mut loaded_state));
        assert_eq!(target_params.gain.unmodulated_plain_value(), 0.75);
        assert!(!target_params.enable.unmodulated_plain_value());
        assert_eq!(&*target_params.sample_path.lock(), "/samples/kick.wav");
    }

    #[test]
    fn preset_state_matches_clap_state_format() {
        // The component state chunk inside a `.vstpreset` file is the same JSON blob the CLAP
        // wrapper stores through its state extension, so presets can be shared between formats
        let params = Arc::new(TestParams::default());
        params.gain.set_plain_value(0.25);
        let state_bytes = serialize_test_state(&params);

        let preset_data = serialize_vstpreset(&TEST_CLASS_ID, &state_bytes);
        let chunk = parse_vstpreset(&preset_data, &TEST_CLASS_ID).unwrap();

        // `deserialize_json()` is exactly what the CLAP wrapper uses to restore state
        let state = unsafe { state::deserialize_json(chunk) }.unwrap();
        assert_eq!(state.version, "2.0.0");
        assert_eq!(
            state.params.get("gain"),
            Some(&state::ParamValue::F32(0.25))
        );
    }

    #[test]
    fn version_migration_is_applied_to_presets() {
        // A preset saved with version 1.0.0 of the plugin, using the old parameter ID
        let old_state_json = br#"{
            "version": "1.0.0",
            "params": {"old_gain": {"f32": 0.9}},
            "fields": {}
        }"#;
        let preset_data = serialize_vstpreset(&TEST_CLASS_ID, old_state_json);
        let mut state = parse_vstpreset(&preset_data, &TEST_CLASS_ID)
            .and_then(|chunk| {
                unsafe { state::deserialize_json(chunk) }
                    .ok_or_else(|| anyhow!("deserialize failed"))
            })
            .unwrap();

        let params = Arc::new(TestParams::default());
        assert!(deserialize_test_state(&params, &mut state));
        // `filter_state()` should have renamed `old_gain` to `gain`
        assert_eq!(params.gain.unmodulated_plain_value(), 0.9);
    }

    #[test]
    fn corrupt_preset_preserves_current_state() {
        let temp = TempDir::new("corrupt");
        let params = Arc::new(TestParams::default());
        params.gain.set_plain_value(0.33);

        // A file that isn't a preset at all
        let garbage_path = temp.path().join("garbage.vstpreset");
        fs::write(&garbage_path, b"this is not a preset").unwrap();
        assert!(load_preset_file(&garbage_path, &TEST_CLASS_ID).is_err());

        // A preset for another plugin
        let other_path = temp.path().join("other.vstpreset");
        fs::write(
            &other_path,
            serialize_vstpreset(&OTHER_CLASS_ID, b"{}"),
        )
        .unwrap();
        assert!(load_preset_file(&other_path, &TEST_CLASS_ID).is_err());

        // A preset with corrupt state data
        let corrupt_state_path = temp.path().join("corrupt-state.vstpreset");
        fs::write(
            &corrupt_state_path,
            serialize_vstpreset(&TEST_CLASS_ID, b"{invalid json"),
        )
        .unwrap();
        assert!(load_preset_file(&corrupt_state_path, &TEST_CLASS_ID).is_err());

        // A truncated preset
        let truncated_path = temp.path().join("truncated.vstpreset");
        let valid = serialize_vstpreset(&TEST_CLASS_ID, &serialize_test_state(&params));
        fs::write(&truncated_path, &valid[..valid.len() / 2]).unwrap();
        assert!(load_preset_file(&truncated_path, &TEST_CLASS_ID).is_err());

        // None of these failures should have touched the current state
        assert_eq!(params.gain.unmodulated_plain_value(), 0.33);
        assert!(params.enable.unmodulated_plain_value());
    }

    #[test]
    fn missing_persisted_field_keeps_current_value() {
        // A preset saved before the `sample_path` field existed (or with a missing sample) should
        // load fine and keep the field's current value
        let state_json = br#"{
            "version": "2.0.0",
            "params": {"gain": {"f32": 0.8}},
            "fields": {}
        }"#;
        let mut state: PluginState = serde_json::from_slice(state_json).unwrap();

        let params = Arc::new(TestParams::default());
        *params.sample_path.lock() = "/samples/current.wav".to_string();
        assert!(deserialize_test_state(&params, &mut state));
        assert_eq!(params.gain.unmodulated_plain_value(), 0.8);
        assert_eq!(&*params.sample_path.lock(), "/samples/current.wav");
    }

    #[test]
    fn unknown_and_mismatched_params_are_skipped() {
        // State containing an unknown parameter and a value of the wrong type should still load
        // the valid parts
        let state_json = br#"{
            "version": "2.0.0",
            "params": {
                "gain": {"f32": 0.6},
                "removed_param": {"f32": 1.0},
                "enable": {"string": "not a bool"}
            },
            "fields": {}
        }"#;
        let mut state: PluginState = serde_json::from_slice(state_json).unwrap();

        let params = Arc::new(TestParams::default());
        assert!(deserialize_test_state(&params, &mut state));
        assert_eq!(params.gain.unmodulated_plain_value(), 0.6);
        // The mismatched value should not have been applied
        assert!(params.enable.unmodulated_plain_value());
    }

    #[test]
    fn concurrent_saves_produce_valid_presets() {
        let temp = TempDir::new("concurrent");
        let user_dir = temp.path().join("user");

        // Multiple threads saving different presets to the same directory
        let mut handles = Vec::new();
        for thread_idx in 0..8 {
            let user_dir = user_dir.clone();
            handles.push(std::thread::spawn(move || {
                for preset_idx in 0..8 {
                    let params = Arc::new(TestParams::default());
                    params.gain.set_plain_value(thread_idx as f32 / 8.0);
                    let state_bytes = serialize_test_state(&params);
                    save_preset_file(
                        &user_dir.join(format!("preset-{thread_idx}-{preset_idx}.vstpreset")),
                        &TEST_CLASS_ID,
                        &state_bytes,
                    )
                    .unwrap();
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }

        // All presets should exist and parse
        let library = PresetLibrary::scan(&[(user_dir.clone(), PresetOrigin::User)]);
        assert_eq!(library.presets().len(), 64);
        for preset in library.presets() {
            let state = load_preset_file(&preset.path, &TEST_CLASS_ID).unwrap();
            assert_eq!(state.version, "2.0.0");
        }

        // Multiple threads saving to the same file concurrently: whatever the final content is,
        // it must be a complete, valid preset (the atomic rename guarantees this)
        let shared_path = user_dir.join("shared.vstpreset");
        let mut handles = Vec::new();
        for thread_idx in 0..8 {
            let shared_path = shared_path.clone();
            handles.push(std::thread::spawn(move || {
                for _ in 0..16 {
                    let params = Arc::new(TestParams::default());
                    params.gain.set_plain_value(thread_idx as f32 / 8.0);
                    let state_bytes = serialize_test_state(&params);
                    save_preset_file(&shared_path, &TEST_CLASS_ID, &state_bytes).unwrap();
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
        let state = load_preset_file(&shared_path, &TEST_CLASS_ID).unwrap();
        assert_eq!(state.version, "2.0.0");
    }

    #[test]
    fn oversized_state_is_rejected() {
        // Input larger than the maximum state size should be rejected before parsing
        let huge = vec![b' '; state::MAX_STATE_BYTES + 1];
        assert!(unsafe { state::deserialize_json(&huge) }.is_none());

        // And saving a state blob larger than the maximum should fail
        let temp = TempDir::new("oversized");
        let path = temp.path().join("huge.vstpreset");
        assert!(save_preset_file(&path, &TEST_CLASS_ID, &huge).is_err());
        assert!(!path.exists());
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn decompression_bombs_are_rejected() {
        // A small compressed blob that expands beyond the state size limit should be rejected
        let huge_json = vec![b' '; state::MAX_STATE_BYTES + 1];
        let compressed = zstd::encode_all(huge_json.as_slice(), 0).unwrap();
        assert!(compressed.len() < state::MAX_STATE_BYTES);
        assert!(unsafe { state::deserialize_json(&compressed) }.is_none());
    }
}
