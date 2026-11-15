//! `demerzel-mock <transcript.json>`: serve the replies of a scripted
//! model transcript as Chat Completions streams, once each, in request
//! order. A test tool for the hermetic smoke runner; never part of the
//! shipped `demerzel` binary.
//!
//! Only `POST <base>/v1/chat/completions` takes a reply; other paths
//! get 404 and other methods 405. A transcript with a `judge` member
//! also serves `POST <base>/judge/v1/systemone`, answered by question
//! id. A judge request that names another model than the `judge`
//! member, or that holds no question, gets 422 and counts as refused.
//! On start the binary prints the server's base URL as one line on
//! stdout. It serves until stdin reaches end of file, then prints
//! `{"served":<n>,"unused":<m>,"refused":<k>}` as a second line and
//! exits 0; with a `judge` member the line is
//! `{"served":<n>,"unused":<m>,"refused":<k>,"judged":<j>}`. `refused`
//! counts the requests that got an error and no reply: 400, 401, 404,
//! 405, 410, and 422. `judged` counts the judge requests that got an
//! answer.

use std::ffi::OsString;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use demerzel_mock::MockServer;
use demerzel_mock::transcript::Transcript;

const USAGE: &str = "usage: demerzel-mock <transcript.json>";

const HELP: &str = "\
Serve the replies of a transcript file as Chat Completions streams.

Each POST to <base>/v1/chat/completions gets the next reply. Other
paths get 404, and other methods get 405. A request after the last
reply gets 410. After a tool_calls reply, the next request needs one
tool message per call id, or it gets 400.

A transcript with a judge member also serves POST
<base>/judge/v1/systemone. Each question gets the answer that the
judge member gives for its id. A request without a bearer key gets
401. A question without an answer, or with an answer of another type,
gets 422. A request that names another model than the judge member,
or that holds no question, gets 422 too.

The first line on stdout is the base URL. The mock stops as soon as
stdin reaches end of file, so a run with no open stdin, such as a
background job, stops at once. At that point the mock prints
{\"served\":<n>,\"unused\":<m>,\"refused\":<k>} on stdout and exits 0.
With a judge member, the line is
{\"served\":<n>,\"unused\":<m>,\"refused\":<k>,\"judged\":<j>}.
The refused count is the number of requests that got an error and no
reply. The judged count is the number of judge requests that got an
answer.";

fn main() -> ExitCode {
    let args: Vec<OsString> = std::env::args_os().skip(1).collect();
    let path = match args.as_slice() {
        [flag] if flag == "-h" || flag == "--help" => {
            println!("{USAGE}\n\n{HELP}");
            return ExitCode::SUCCESS;
        }
        [path] if !path.to_string_lossy().starts_with('-') => PathBuf::from(path),
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match load(&path) {
        Ok(transcript) => serve(transcript),
        Err(reason) => {
            eprintln!("demerzel-mock: {}: {reason}", path.display());
            ExitCode::FAILURE
        }
    }
}

fn load(path: &Path) -> Result<Transcript, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    Transcript::parse(&text).map_err(|e| e.to_string())
}

#[tokio::main(flavor = "current_thread")]
async fn serve(transcript: Transcript) -> ExitCode {
    let total = transcript.replies.len() as u64;
    let server = MockServer::start_chat(transcript.replies).await;
    let judged = transcript.judge.is_some();
    if let Some(judge) = transcript.judge {
        server.set_judge(judge.script);
    }
    if print_line(&server.url()).is_err() {
        return ExitCode::FAILURE;
    }
    // A read error counts as end of file: the runner is gone either way.
    let (eof, closed) = tokio::sync::oneshot::channel::<()>();
    std::thread::spawn(move || {
        let _ = std::io::copy(&mut std::io::stdin().lock(), &mut std::io::sink());
        let _ = eof.send(());
    });
    let _ = closed.await;
    let served = server.served();
    // Built by hand: the key order is part of the line's contract.
    let mut summary = format!(
        r#"{{"served":{served},"unused":{},"refused":{}"#,
        total.saturating_sub(served),
        server.refused()
    );
    if judged {
        summary.push_str(&format!(r#","judged":{}"#, server.judged()));
    }
    summary.push('}');
    match print_line(&summary) {
        Ok(()) => ExitCode::SUCCESS,
        Err(_) => ExitCode::FAILURE,
    }
}

/// Write one line to stdout and flush it.
fn print_line(line: &str) -> std::io::Result<()> {
    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "{line}")?;
    stdout.flush()
}
